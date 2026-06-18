from __future__ import annotations
import json
import os
import time
import urllib.error
import urllib.request
from dotenv import load_dotenv
from .prompts import ACTOR_SYSTEM, EVALUATOR_SYSTEM, REFLECTOR_SYSTEM
from .schemas import JudgeResult, QAExample, ReflectionEntry
from .utils import normalize_answer

load_dotenv()


def actor_answer(example: QAExample, attempt_id: int, agent_type: str, reflection_memory: list[str]) -> str:
    context = _format_context(example)
    reflections = "\n".join(f"- {item}" for item in reflection_memory) or "None"
    user = f"""
Question: {example.question}

Context:
{context}

Attempt: {attempt_id}
Agent type: {agent_type}
Reflection memory:
{reflections}

Return only the final answer text.
"""
    content, _usage, _latency_ms = _chat(
        [
            {"role": "system", "content": ACTOR_SYSTEM},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
    )
    return content.strip().strip('"')

def evaluator(example: QAExample, answer: str) -> JudgeResult:
    user = f"""
Question: {example.question}
Gold answer: {example.gold_answer}
Predicted answer: {answer}

Evaluate the predicted answer.
"""
    content, _usage, _latency_ms = _chat(
        [
            {"role": "system", "content": EVALUATOR_SYSTEM},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    payload = _loads_json_object(content)
    if payload is None:
        score = int(normalize_answer(example.gold_answer) == normalize_answer(answer))
        return JudgeResult(
            score=score,
            reason="Evaluator returned invalid JSON; fell back to normalized exact match.",
            missing_evidence=[] if score else ["Unable to parse evaluator feedback."],
            spurious_claims=[] if score else [answer],
        )
    return JudgeResult.model_validate(payload)

def reflector(example: QAExample, attempt_id: int, judge: JudgeResult) -> ReflectionEntry:
    user = f"""
Question: {example.question}
Gold answer: {example.gold_answer}
Failed attempt id: {attempt_id}
Evaluator reason: {judge.reason}
Missing evidence: {judge.missing_evidence}
Spurious claims: {judge.spurious_claims}

Create a reflection for the next attempt.
"""
    content, _usage, _latency_ms = _chat(
        [
            {"role": "system", "content": REFLECTOR_SYSTEM},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    payload = _loads_json_object(content)
    if payload is None:
        return ReflectionEntry(
            attempt_id=attempt_id,
            failure_reason=judge.reason,
            lesson="Use the evaluator feedback before answering again.",
            next_strategy="Re-read the relevant context and verify the final answer against the gold task requirements.",
        )
    payload["attempt_id"] = attempt_id
    return ReflectionEntry.model_validate(payload)

def _chat(
    messages: list[dict[str, str]],
    temperature: float = 0.0,
    response_format: dict | None = None,
) -> tuple[str, dict, int]:
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY or LLM_API_KEY before running with --mode llm.")

    base_url = (os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_BASE_URL")).rstrip("/")
    model = os.getenv("OPENAI_MODEL") or os.getenv("LLM_MODEL")
    body = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if response_format is not None:
        body["response_format"] = response_format

    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM request failed with HTTP {exc.code}: {detail}") from exc
    latency_ms = round((time.perf_counter() - started) * 1000)
    message = payload["choices"][0]["message"]["content"]
    return message, payload.get("usage", {}), latency_ms

def _format_context(example: QAExample) -> str:
    return "\n\n".join(f"[{chunk.title}]\n{chunk.text}" for chunk in example.context)

def _loads_json_object(text: str) -> dict | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
