#!/usr/bin/env python3
"""AI Prompt Evaluation System — CLI

Commands
--------
run      — execute an evaluation experiment
report   — print results for an existing experiment
list     — list all experiments
compare  — side-by-side comparison of two experiments

Quick start (Groq — free)
--------------------------
1. Get a free key at console.groq.com
2. Add it to .env:  GROQ_API_KEY=gsk_...
3. Run:
   python cli.py run -n "baseline" \\
     -p prompts/v1_basic.yaml \\
     -d datasets/banking_assistant_tests.json
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import typer
from dotenv import load_dotenv
from rich.console import Console

load_dotenv()

app = typer.Typer(
    help="AI Prompt Evaluation System — evaluate prompts across LLM providers.",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()

_PROVIDERS = ["groq", "claude"]


@app.command()
def run(
    name: str = typer.Option(..., "--name", "-n", help="Experiment name"),
    prompts: List[Path] = typer.Option(
        ..., "--prompt", "-p",
        help="Prompt YAML file (repeat for multiple prompts)",
    ),
    dataset: Path = typer.Option(
        ..., "--dataset", "-d",
        help="Test-cases JSON file",
    ),
    provider: str = typer.Option(
        "groq", "--provider",
        help="Generation provider: groq (free) | claude",
    ),
    runner_model: Optional[str] = typer.Option(
        None, "--runner-model",
        help="Override default model (groq default: llama-3.3-70b-versatile)",
    ),
    judge_model: str = typer.Option(
        "claude-haiku-4-5-20251001", "--judge-model",
        help="Claude model for LLM judge (auto-falls back to groq/llama-3.1-8b-instant if no Anthropic key)",
    ),
    description: str = typer.Option("", "--desc", help="Free-text description"),
    no_judge: bool = typer.Option(
        False, "--no-judge",
        help="Skip LLM judge entirely",
    ),
    db: str = typer.Option("experiments/results.db", "--db", help="SQLite DB path"),
    anthropic_key: Optional[str] = typer.Option(
        None, "--anthropic-key", envvar="ANTHROPIC_API_KEY",
        help="Anthropic API key (optional — for LLM judge or Claude runner)",
    ),
    groq_key: Optional[str] = typer.Option(
        None, "--groq-key", envvar="GROQ_API_KEY",
        help="Groq API key — free at console.groq.com",
    ),
) -> None:
    """Run a prompt evaluation experiment."""
    if provider not in _PROVIDERS:
        console.print(f"[red]Unknown provider '{provider}'. Choose: {_PROVIDERS}[/red]")
        raise typer.Exit(1)

    missing = [str(p) for p in prompts if not p.exists()]
    if missing:
        console.print(f"[red]Prompt file(s) not found: {missing}[/red]")
        raise typer.Exit(1)
    if not dataset.exists():
        console.print(f"[red]Dataset not found: {dataset}[/red]")
        raise typer.Exit(1)

    from src.pipeline.eval_pipeline import EvaluationPipeline
    from src.reporting.reporter import ExperimentReporter
    from src.storage.db import ExperimentDB

    runner_key = groq_key if provider == "groq" else anthropic_key

    pipeline = EvaluationPipeline(
        provider=provider,
        runner_model=runner_model,
        judge_model=judge_model,
        db_path=db,
        api_key=runner_key,
        anthropic_api_key=anthropic_key,
        groq_api_key=groq_key,
        no_judge=no_judge,
    )
    experiment_id = pipeline.run(
        name=name,
        prompt_paths=[str(p) for p in prompts],
        dataset_path=str(dataset),
        description=description,
    )

    reporter = ExperimentReporter(ExperimentDB(db_path=db))
    reporter.print_summary(experiment_id)


@app.command()
def report(
    experiment_id: str = typer.Argument(..., help="Experiment ID to show"),
    export: Optional[Path] = typer.Option(
        None, "--export", "-e",
        help="Export results to JSON file",
    ),
    db: str = typer.Option("experiments/results.db", "--db"),
) -> None:
    """Print a detailed report for an existing experiment."""
    from src.reporting.reporter import ExperimentReporter
    from src.storage.db import ExperimentDB

    reporter = ExperimentReporter(ExperimentDB(db_path=db))
    reporter.print_summary(experiment_id)
    if export:
        reporter.export_json(experiment_id, str(export))


@app.command(name="list")
def list_experiments(
    db: str = typer.Option("experiments/results.db", "--db"),
) -> None:
    """List all recorded experiments."""
    from rich.table import Table
    from src.storage.db import ExperimentDB

    exps = ExperimentDB(db_path=db).list_experiments()
    if not exps:
        console.print("[yellow]No experiments found.[/yellow]")
        return

    table = Table(title="Experiments", border_style="blue")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Name", style="bold")
    table.add_column("Status")
    table.add_column("Provider")
    table.add_column("Created")

    for row in exps:
        color = "green" if row[4] == "completed" else "yellow"
        meta = row[5] if isinstance(row[5], dict) else {}
        provider_tag = meta.get("provider", "—")
        table.add_row(
            row[0], row[1],
            f"[{color}]{row[4]}[/{color}]",
            provider_tag,
            row[3][:19],
        )
    console.print(table)


@app.command()
def compare(
    exp1: str = typer.Argument(..., help="First experiment ID"),
    exp2: str = typer.Argument(..., help="Second experiment ID"),
    db: str = typer.Option("experiments/results.db", "--db"),
) -> None:
    """Side-by-side comparison of two experiments."""
    from src.reporting.reporter import ExperimentReporter
    from src.storage.db import ExperimentDB

    reporter = ExperimentReporter(ExperimentDB(db_path=db))
    console.rule(f"[bold cyan]Experiment: {exp1}[/bold cyan]")
    reporter.print_summary(exp1)
    console.rule(f"[bold cyan]Experiment: {exp2}[/bold cyan]")
    reporter.print_summary(exp2)


if __name__ == "__main__":
    app()
