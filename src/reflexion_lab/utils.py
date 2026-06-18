from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Iterable
from .schemas import QAExample, RunRecord

def normalize_answer(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text

def load_dataset(path: str | Path) -> list[QAExample]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [_load_example(item, index) for index, item in enumerate(raw)]

def _load_example(item: dict, index: int) -> QAExample:
    if "qid" in item and "gold_answer" in item:
        return QAExample.model_validate(item)

    if "_id" in item and "answer" in item and "context" in item:
        context = []
        for title, sentences in item["context"]:
            context.append({"title": title, "text": " ".join(sentences)})
        level = item.get("level", "medium")
        difficulty = level if level in {"easy", "medium", "hard"} else "medium"
        return QAExample.model_validate(
            {
                "qid": item.get("_id") or f"hotpot_{index}",
                "difficulty": difficulty,
                "question": item["question"],
                "gold_answer": item["answer"],
                "context": context,
            }
        )

    raise ValueError(f"Unsupported dataset item format at index {index}")

def save_jsonl(path: str | Path, records: Iterable[RunRecord]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(record.model_dump_json() + "\n")
