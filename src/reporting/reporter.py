from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..storage.db import ExperimentDB

console = Console()


class ExperimentReporter:
    """Render evaluation results as Rich tables; optionally export to JSON."""

    def __init__(self, db: ExperimentDB) -> None:
        self.db = db

    def print_summary(self, experiment_id: str) -> None:
        results = self.db.get_results(experiment_id)
        if not results:
            console.print(f"[yellow]No results found for experiment '{experiment_id}'.[/yellow]")
            return

        by_prompt: Dict[str, List[Dict[str, Any]]] = {}
        for r in results:
            by_prompt.setdefault(r["prompt_name"], []).append(r)

        for prompt_name, rows in by_prompt.items():
            avg_score = sum(r["overall_score"] or 0.0 for r in rows) / len(rows)
            avg_latency = sum(r["latency_ms"] or 0.0 for r in rows) / len(rows)
            judge_used = any(
                any(d["name"] in ("relevance", "accuracy", "safety") for d in r["dimensions"])
                for r in rows
            )
            judge_tag = "[green]judge+rules[/green]" if judge_used else "[dim]rules only[/dim]"

            table = Table(
                title=(
                    f"[bold]{prompt_name}[/bold]  "
                    f"avg={avg_score:.3f}  latency={avg_latency:.0f}ms  {judge_tag}"
                ),
                border_style="blue",
                show_lines=False,
            )
            table.add_column("Test Case", style="dim", width=12)
            table.add_column("Score", justify="right", width=7)
            table.add_column("Latency", justify="right", width=9)
            table.add_column("Tokens", justify="right", width=7)
            table.add_column("Dimensions", min_width=40)

            _ABBR = {
                "length_appropriateness": "len",
                "format_compliance": "fmt",
                "topic_coverage": "topics",
                "no_red_flags": "flags",
                "relevance": "relev",
                "accuracy": "accur",
                "completeness": "compl",
                "clarity": "clarity",
                "safety": "safety",
            }
            for r in rows:
                score = r["overall_score"] or 0.0
                color = "green" if score >= 0.7 else ("yellow" if score >= 0.5 else "red")
                dims = r["dimensions"]
                dim_str = "  ".join(
                    f"{_ABBR.get(d['name'], d['name'][:6])}={d['score']:.2f}"
                    for d in dims
                ) if dims else "—"
                table.add_row(
                    r["test_case_id"],
                    f"[{color}]{score:.3f}[/{color}]",
                    f"{r['latency_ms']:.0f}ms",
                    str((r["input_tokens"] or 0) + (r["output_tokens"] or 0)),
                    dim_str,
                )
            console.print(table)
            console.print()

    def export_json(self, experiment_id: str, output_path: str) -> None:
        results = self.db.get_results(experiment_id)
        Path(output_path).write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        console.print(f"[green]Exported {len(results)} results → {output_path}[/green]")
