from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
from .schemas import AttemptTrace, QAExample, ReflectionEntry, RunRecord

RuntimeMode = Literal["mock", "llm"]

@dataclass
class BaseAgent:
    agent_type: Literal["react", "reflexion"]
    max_attempts: int = 1
    runtime: RuntimeMode = "llm"

    def run(self, example: QAExample) -> RunRecord:
        runtime_module = self._runtime_module()
        reflection_memory: list[str] = []
        reflections: list[ReflectionEntry] = []
        traces: list[AttemptTrace] = []
        final_answer = ""
        final_score = 0

        for attempt_id in range(1, self.max_attempts + 1):
            answer = runtime_module.actor_answer(example, attempt_id, self.agent_type, reflection_memory)
            judge = runtime_module.evaluator(example, answer)
            # TODO: Replace with actual token count from LLM response.
            token_estimate = 320 + (attempt_id * 65) + (120 if self.agent_type == "reflexion" else 0)
            # TODO: Replace with actual latency measurement.
            latency_ms = 160 + (attempt_id * 40) + (90 if self.agent_type == "reflexion" else 0)

            trace = AttemptTrace(
                attempt_id=attempt_id,
                answer=answer,
                score=judge.score,
                reason=judge.reason,
                token_estimate=token_estimate,
                latency_ms=latency_ms,
            )
            final_answer = answer
            final_score = judge.score

            if judge.score == 1:
                traces.append(trace)
                break

            if self.agent_type == "reflexion" and attempt_id < self.max_attempts:
                reflection = runtime_module.reflector(example, attempt_id, judge)
                reflections.append(reflection)
                trace.reflection = reflection
                reflection_memory.append(
                    f"Attempt {attempt_id} failed: {reflection.failure_reason} "
                    f"Lesson: {reflection.lesson} Next strategy: {reflection.next_strategy}"
                )

            traces.append(trace)

        total_tokens = sum(t.token_estimate for t in traces)
        total_latency = sum(t.latency_ms for t in traces)
        failure_modes = getattr(runtime_module, "FAILURE_MODE_BY_QID", {})
        failure_mode = "none" if final_score == 1 else failure_modes.get(example.qid, "wrong_final_answer")
        return RunRecord(
            qid=example.qid,
            question=example.question,
            gold_answer=example.gold_answer,
            agent_type=self.agent_type,
            predicted_answer=final_answer,
            is_correct=bool(final_score),
            attempts=len(traces),
            token_estimate=total_tokens,
            latency_ms=total_latency,
            failure_mode=failure_mode,
            reflections=reflections,
            traces=traces,
        )

    def _runtime_module(self):
        if self.runtime == "llm":
            from . import llm_runtime

            return llm_runtime
        if self.runtime == "mock":
            from . import mock_runtime

            return mock_runtime
        raise ValueError(f"Unsupported runtime: {self.runtime}")

class ReActAgent(BaseAgent):
    def __init__(self, runtime: RuntimeMode = "llm") -> None:
        super().__init__(agent_type="react", max_attempts=1, runtime=runtime)

class ReflexionAgent(BaseAgent):
    def __init__(self, max_attempts: int = 3, runtime: RuntimeMode = "llm") -> None:
        super().__init__(agent_type="reflexion", max_attempts=max_attempts, runtime=runtime)
