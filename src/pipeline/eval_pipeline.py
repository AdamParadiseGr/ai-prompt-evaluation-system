from __future__ import annotations

import json
import uuid
from typing import List, Literal, Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

from ..evaluators.base_judge import BaseJudge
from ..evaluators.rule_based import RuleBasedEvaluator
from ..models.evaluation import EvaluationResult, TestCase
from ..models.prompt import PromptVersion
from ..runners.base import BaseRunner
from ..storage.db import ExperimentDB

console = Console()

Provider = Literal["groq", "claude"]


def _build_runner(
    provider: Provider,
    model: Optional[str],
    api_key: Optional[str],
) -> BaseRunner:
    if provider == "groq":
        from ..runners.groq_runner import GroqRunner
        return GroqRunner(model=model or "llama-3.3-70b-versatile", api_key=api_key)
    from ..runners.claude_runner import ClaudeRunner
    return ClaudeRunner(model=model or "claude-haiku-4-5-20251001", api_key=api_key)


def _build_judge(
    anthropic_key: Optional[str],
    groq_key: Optional[str],
    judge_model: str,
    no_judge: bool = False,
) -> Optional[BaseJudge]:
    if no_judge:
        return None
    # Claude judge (если есть реальный ключ)
    if anthropic_key and anthropic_key != "placeholder":
        from ..evaluators.claude_judge import ClaudeJudge
        return ClaudeJudge(model=judge_model, api_key=anthropic_key)
    # Groq judge — пробуем всегда: синглтон уже создан runner'ом,
    # api_key=None корректно вернёт существующий инстанс
    try:
        from ..evaluators.groq_judge import GroqJudge
        groq_judge_model = judge_model if not judge_model.startswith("claude") else "llama-3.3-70b-versatile"
        return GroqJudge(model=groq_judge_model, api_key=groq_key)
    except EnvironmentError:
        return None


class EvaluationPipeline:
    """Orchestrates a full prompt-evaluation experiment.

    Flow: load prompts → load test cases → for each pair:
      runner (Groq/Claude) → rule-based checks → LLM judge → persist to SQLite
    """

    def __init__(
        self,
        provider: Provider = "groq",
        runner_model: Optional[str] = None,
        judge_model: str = "claude-haiku-4-5-20251001",
        db_path: str = "experiments/results.db",
        api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        groq_api_key: Optional[str] = None,
        no_judge: bool = False,
    ) -> None:
        self.runner: BaseRunner = _build_runner(provider, runner_model, api_key)
        effective_groq_key = groq_api_key or (api_key if provider == "groq" else None)
        self.judge: Optional[BaseJudge] = _build_judge(
            anthropic_api_key, effective_groq_key, judge_model, no_judge
        )
        self.rule_checker = RuleBasedEvaluator()
        self.db = ExperimentDB(db_path=db_path)
        self.provider = provider

    def load_test_cases(self, dataset_path: str) -> List[TestCase]:
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [TestCase(**tc) for tc in data["test_cases"]]

    def load_prompts(self, prompt_paths: List[str]) -> List[PromptVersion]:
        return [PromptVersion.from_yaml(p) for p in prompt_paths]

    def run(
        self,
        name: str,
        prompt_paths: List[str],
        dataset_path: str,
        description: str = "",
    ) -> str:
        """Execute evaluation experiment. Returns experiment_id."""
        experiment_id = str(uuid.uuid4())[:8]
        self.db.create_experiment(
            experiment_id, name, description,
            metadata={"provider": self.provider},
        )

        prompts = self.load_prompts(prompt_paths)
        test_cases = self.load_test_cases(dataset_path)
        total = len(prompts) * len(test_cases)

        if self.judge is None:
            judge_status = "[dim]skipped (no key available)[/dim]"
        elif type(self.judge).__name__ == "GroqJudge":
            judge_status = f"[cyan]groq/{self.judge.model}[/cyan]"
        else:
            judge_status = f"[green]{self.judge.model}[/green]"

        console.print(
            Panel.fit(
                f"[bold cyan]Experiment:[/bold cyan] [bold]{name}[/bold]\n"
                f"[dim]ID:[/dim] {experiment_id}  "
                f"[dim]Provider:[/dim] {self.provider}  "
                f"[dim]Prompts:[/dim] {len(prompts)}  "
                f"[dim]Cases:[/dim] {len(test_cases)}  "
                f"[dim]Total:[/dim] {total}\n"
                f"[dim]LLM judge:[/dim] {judge_status}",
                border_style="cyan",
                title="Starting evaluation",
            )
        )

        errors = 0
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Running...", total=total)
            for prompt in prompts:
                for tc in test_cases:
                    progress.update(
                        task,
                        description=f"[cyan]{prompt.name}[/cyan] / [dim]{tc.id}[/dim]",
                    )
                    try:
                        result = self._evaluate_one(prompt, tc, experiment_id)
                        self.db.save_result(result)
                    except Exception as exc:
                        console.print(f"  [red]✗ {prompt.name}/{tc.id}: {exc}[/red]")
                        errors += 1
                    progress.advance(task)

        self.db.complete_experiment(experiment_id)
        status = (
            "[bold green]Completed[/bold green]"
            if errors == 0
            else f"[bold yellow]Completed with {errors} error(s)[/bold yellow]"
        )
        console.print(f"\n{status} — experiment [bold]{experiment_id}[/bold]")
        return experiment_id

    def _evaluate_one(
        self,
        prompt: PromptVersion,
        tc: TestCase,
        experiment_id: str,
    ) -> EvaluationResult:
        response_text, latency_ms, in_tok, out_tok = self.runner.run(prompt, tc)

        dimensions = self.rule_checker.evaluate(tc, response_text, prompt.name)
        if self.judge is not None:
            dimensions.extend(self.judge.evaluate(tc, response_text))

        return EvaluationResult(
            experiment_id=experiment_id,
            test_case_id=tc.id,
            prompt_version=prompt.version,
            prompt_name=prompt.name,
            raw_response=response_text,
            latency_ms=latency_ms,
            input_tokens=in_tok,
            output_tokens=out_tok,
            dimensions=dimensions,
            metadata={
                "question": tc.question,
                "difficulty": tc.difficulty,
                "tags": tc.tags,
                "provider": self.provider,
            },
        )
