from __future__ import annotations
import json
from pathlib import Path
import typer
from rich import print
from src.reflexion_lab.agents import ReActAgent, ReflexionAgent
from src.reflexion_lab.reporting import build_report, save_report
from src.reflexion_lab.schemas import RunRecord
from src.reflexion_lab.utils import load_dataset, save_jsonl

app = typer.Typer(add_completion=False)

@app.command()
def main(
    dataset: str = "data/hotpot_dev_distractor_v1.jsonl",
    out_dir: str = "outputs/llm_runs",
    reflexion_attempts: int = 3,
    mode: str = "llm",
    agent: str = "both",
    batch_size: int = 10,
    limit: int = 50,
    stop_on_error: bool = False,
) -> None:
    if mode not in {"mock", "llm"}:
        raise typer.BadParameter("mode must be 'mock' or 'llm'")
    if agent not in {"react", "reflexion", "both"}:
        raise typer.BadParameter("agent must be 'react', 'reflexion', or 'both'")
    if batch_size < 1:
        raise typer.BadParameter("batch_size must be >= 1")
    if limit < 1:
        raise typer.BadParameter("limit must be >= 1")

    examples = load_dataset(dataset)[:limit]
    print(f"[cyan]Loaded[/cyan] {len(examples)} question(s) from {dataset}")
    out_path = Path(out_dir)
    all_records: list[RunRecord] = []

    if agent in {"react", "both"}:
        react = ReActAgent(runtime=mode)
        react_records = _run_in_batches("react", react, examples, batch_size, stop_on_error)
        save_jsonl(out_path / "react_runs.jsonl", react_records)
        all_records.extend(react_records)

    if agent in {"reflexion", "both"}:
        reflexion = ReflexionAgent(max_attempts=reflexion_attempts, runtime=mode)
        reflexion_records = _run_in_batches("reflexion", reflexion, examples, batch_size, stop_on_error)
        save_jsonl(out_path / "reflexion_runs.jsonl", reflexion_records)
        all_records.extend(reflexion_records)

    report = build_report(all_records, dataset_name=Path(dataset).name, mode=mode)
    json_path, md_path = save_report(report, out_path)
    print(f"[green]Saved[/green] {json_path}")
    print(f"[green]Saved[/green] {md_path}")
    print(json.dumps(report.summary, indent=2))

def _run_in_batches(agent_name: str, agent_obj, examples, batch_size: int, stop_on_error: bool) -> list[RunRecord]:
    records: list[RunRecord] = []
    for start in range(0, len(examples), batch_size):
        end = min(start + batch_size, len(examples))
        batch = examples[start:end]
        print(f"[cyan]{agent_name}[/cyan] scoring questions {start + 1}-{end}")
        batch_records = [agent_obj.run(example) for example in batch]
        records.extend(batch_records)

        wrong = [record for record in batch_records if not record.is_correct]
        if wrong:
            print(f"[yellow]{agent_name}[/yellow] found {len(wrong)} wrong answer(s) in this batch.")
        elif stop_on_error:
            print(f"[green]{agent_name}[/green] batch has no wrong answers; continuing.")

        if stop_on_error and wrong:
            first_wrong = wrong[0]
            print(
                f"[red]{agent_name}[/red] stopping at batch {start + 1}-{end}; "
                f"first wrong qid={first_wrong.qid}, predicted={first_wrong.predicted_answer!r}, "
                f"gold={first_wrong.gold_answer!r}"
            )
            break

    return records

if __name__ == "__main__":
    app()
