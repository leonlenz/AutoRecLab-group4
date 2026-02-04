from __future__ import annotations

import json
import re
from dataclasses import is_dataclass, fields as dc_fields
from typing import Any, Iterable, Type


class StructuredRecoveryPolicy:
    def __init__(
        self,
        max_attempts: int = 3,
        bad_markers: Iterable[str] | None = None,
        incoherent_len_threshold: int = 400,
    ) -> None:
        self.max_attempts = max_attempts
        self.bad_markers = list(
            bad_markers
            if bad_markers is not None
            else ["commentary to=", "<|", "functions.", "}]**", "ListToolsRequest"]
        )
        self.incoherent_len_threshold = incoherent_len_threshold


def _schema_instructions(schema: Type[Any]) -> str:
    if is_dataclass(schema):
        lines = []
        for f in dc_fields(schema):
            tname = getattr(f.type, "__name__", str(f.type))
            lines.append(f'- "{f.name}": {tname}')
        return "JSON object with exactly these keys:\n" + "\n".join(lines)

    # Pydantic v1 model fallback
    if hasattr(schema, "__fields__"):
        lines = []
        for name, f in schema.__fields__.items():  # type: ignore[attr-defined]
            tname = getattr(getattr(f, "type_", None), "__name__", "any")
            lines.append(f'- "{name}": {tname}')
        return "JSON object with exactly these keys:\n" + "\n".join(lines)

    return f"JSON object matching schema type {getattr(schema, '__name__', str(schema))}."


def _strip_fences(text: str) -> str:
    text = (text or "").strip()
    return re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text, flags=re.IGNORECASE).strip()


def _coerce_faulty_output(agent_response: Any) -> str:
    if agent_response is None:
        return ""

    if isinstance(agent_response, dict):
        if "structured_response" in agent_response:
            return str(agent_response["structured_response"])

        msgs = agent_response.get("messages")
        if msgs:
            last = msgs[-1]
            return str(getattr(last, "content", last))
        return str(agent_response)

    return str(getattr(agent_response, "content", agent_response))


def _is_empty_or_incoherent(text: str, policy: StructuredRecoveryPolicy) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    if any(m in t for m in policy.bad_markers) and len(t) < policy.incoherent_len_threshold:
        return True
    return False


def _parse_json_object(text: str) -> dict:
    text = _strip_fences(text)
    try:
        obj = json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            raise
        obj = json.loads(m.group(0))

    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object, got {type(obj)}")
    return obj


def _allowed_keys(schema: Type[Any]) -> set[str]:
    if is_dataclass(schema):
        return {f.name for f in dc_fields(schema)}
    if hasattr(schema, "__fields__"):  # pydantic v1
        return set(schema.__fields__.keys())  # type: ignore[attr-defined]
    return set()


def structured_output(
    *,
    llm: Any,
    schema: Type[Any],
    task_prompt: str,
    policy: StructuredRecoveryPolicy,
) -> Any:
    schema_text = _schema_instructions(schema)

    last_text = ""
    last_err = ""

    for attempt in range(policy.max_attempts + 1):
        if attempt == 0:
            cur_prompt = f"""
Return ONLY a single JSON object. No markdown, no prose, no backticks.

TARGET STRUCTURE:
{schema_text}

TASK:
{task_prompt}
""".strip()
        else:
            regen = _is_empty_or_incoherent(last_text, policy)
            instruction = (
                "The previous output is empty/incoherent. Regenerate from scratch from the TASK."
                if regen
                else "Convert the previous output into the TARGET STRUCTURE JSON."
            )

            cur_prompt = f"""
Return ONLY a single JSON object. No markdown, no prose, no backticks.

TARGET STRUCTURE:
{schema_text}

TASK (reference):
{task_prompt}

PREVIOUS OUTPUT (convert if usable):
{last_text}

PARSING ERROR:
{last_err}

INSTRUCTION:
{instruction}

Now return ONLY the corrected JSON object:
""".strip()

        resp = llm.invoke(cur_prompt)
        text = str(getattr(resp, "content", resp))
        last_text = text

        try:
            data = _parse_json_object(text)
            allowed = _allowed_keys(schema)
            filtered = {k: v for k, v in data.items() if (k in allowed) or not allowed}
            return schema(**filtered)
        except Exception as e:
            last_err = repr(e)

    raise ValueError(
        f"Failed to produce valid structured output for {getattr(schema, '__name__', str(schema))} "
        f"after {policy.max_attempts + 1} attempts. Last error: {last_err}. Last output: {last_text}"
    )


def ensure_structured_agent_response(
    *,
    agent_response: Any,
    schema: Type[Any],
    llm: Any,
    original_prompt: str,
    policy: StructuredRecoveryPolicy,
) -> Any:
    faulty_output = _coerce_faulty_output(agent_response)

    repair_task = f"""
You must produce output for the original task, formatted as the target JSON structure.

Original task:
{original_prompt}

Faulty output (use if helpful, otherwise regenerate):
{faulty_output}
""".strip()

    return structured_output(llm=llm, schema=schema, task_prompt=repair_task, policy=policy)
