"""Automatically fill metadata gaps using Google ADK agents."""
from __future__ import annotations

import argparse
import asyncio
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, cast

from clm_with_chromadb import (
    ContractVectorDB,
    GEMINI_MODEL_NAME,
    LOCAL_LLM_AVAILABLE,
    LOCAL_MODEL_NAME,
    MODEL_PROVIDER,
    LocalOllamaLlm,
)
from metadata_gap_scanner import (
    HIGH_VALUE_FIELDS,
    build_adk_prompt,
    coerce_text,
    collect_gap_data,
    value_missing,
)

try:
    from google.adk.agents import Agent
    from google.adk.models import Gemini
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai.types import Content, Part
    from google.genai.errors import ClientError
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    raise SystemExit("google-adk (and google-genai) must be installed to run metadata_gap_runner.py") from exc


LOCAL_PROVIDER_FLAGS = {"local", "ollama", "onprem", "offline"}


def _coerce_metadata_value(value: Any) -> Any:
    """Convert metadata values to Chroma-safe scalars."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (list, tuple)):
        iterable_value = cast(Iterable[Any], value)
        list_value: List[Any] = list(iterable_value)
        # Represent lists as JSON strings to preserve structure without nested metadata objects.
        try:
            return json.dumps(list_value)
        except (TypeError, ValueError):
            return str(list_value)
    if isinstance(value, dict):
        dict_value = cast(Dict[Any, Any], value)
        try:
            return json.dumps(dict_value, sort_keys=True)
        except (TypeError, ValueError):
            return str(dict_value)
    return str(value)


def _build_runner_llm(cloud_model: Optional[str], local_model: Optional[str]) -> Any:
    wants_local = (MODEL_PROVIDER in LOCAL_PROVIDER_FLAGS) or bool(local_model)
    if wants_local and LOCAL_LLM_AVAILABLE:
        profile = local_model or LOCAL_MODEL_NAME or "ollama"
        if local_model:
            print(f"ℹ️  metadata_gap_runner using local model override '{profile}'.")
        elif LOCAL_MODEL_NAME:
            print(f"ℹ️  metadata_gap_runner using configured local model '{profile}'.")
        else:
            print("ℹ️  metadata_gap_runner using default local Ollama profile.")
        return LocalOllamaLlm(model=profile)

    if wants_local and not LOCAL_LLM_AVAILABLE:
        print("⚠️  Local Ollama adapter unavailable; falling back to Gemini for metadata gap runner.")

    resolved_model = cloud_model or GEMINI_MODEL_NAME
    print(f"ℹ️  metadata_gap_runner using Gemini model '{resolved_model}'.")
    return Gemini(model=resolved_model)


class MetadataGapAgent:
    """Lightweight ADK agent focused on metadata repair prompts."""

    def __init__(self, llm_model: Any) -> None:
        self.session_service = InMemorySessionService()
        self.agent = Agent(
            name="metadata_gap_filler",
            model=llm_model,
            description="Spectralink contracts specialist for metadata repairs",
            instruction=(
                "You are a senior contracts specialist supporting Spectralink's CLM clean-up. "
                "Respect document hierarchies: master agreements govern commercial terms, while SOWs, amendments, and schedules can only override specific clauses they restate. "
                "Respond with STRICT JSON output (no prose, no code fences). Always include 'contract_id', 'confidence' (high/medium/low), and optional 'notes'. "
                "If the text says a field defers to a master agreement, cite that relationship in 'notes' instead of fabricating values. "
                "When you must stream multiple JSON snippets, emit each as a standalone JSON object—the tooling merges them."
            ),
        )
        runner_app_name = getattr(self.agent, "app_name", "clm_metadata_gap")
        self.runner = Runner(
            app_name=runner_app_name,
            agent=self.agent,
            session_service=self.session_service,
        )
        self.session = asyncio.run(
            self.session_service.create_session(
                app_name="clm_metadata_gap",
                user_id="metadata_gap_runner",
            )
        )

    def run_prompt(self, prompt: str) -> str:
        message = Content(role="user", parts=[Part(text=prompt)])
        response_chunks: List[str] = []
        try:
            response_stream = self.runner.run(
                session_id=self.session.id,
                user_id=self.session.user_id,
                new_message=message,
            )

            for event in response_stream:
                content = getattr(event, "content", None)
                if not content:
                    continue
                for part in getattr(content, "parts", []) or []:
                    text = getattr(part, "text", None)
                    if text:
                        response_chunks.append(text)
            return "".join(response_chunks).strip()
        except ClientError as err:
            raise RuntimeError(f"Gemini API error: {err}") from err
        except Exception as err:
            raise RuntimeError(f"ADK runner failure: {err}") from err


def _extract_json_candidate(text: str) -> Optional[str]:
    """Pull a JSON blob out of a raw LLM response."""
    if not text:
        return None

    fenced_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if fenced_match:
        return fenced_match.group(1)

    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        return brace_match.group(0)

    return text if text.strip().startswith("{") and text.strip().endswith("}") else None


def _decode_json_objects(text: str) -> List[Dict[str, Any]]:
    if not text:
        return []
    decoder = json.JSONDecoder()
    idx = 0
    length = len(text)
    objects: List[Dict[str, Any]] = []
    while idx < length:
        char = text[idx]
        if char.isspace():
            idx += 1
            continue
        if char != "{" and char != "[":
            idx += 1
            continue
        try:
            obj, end = decoder.raw_decode(text, idx)
        except json.JSONDecodeError:
            idx += 1
            continue
        idx = end
        if isinstance(obj, dict):
            objects.append(cast(Dict[str, Any], obj))
            continue
        if isinstance(obj, list):
            for item in cast(List[Any], obj):
                if isinstance(item, dict):
                    objects.append(cast(Dict[str, Any], item))
    return objects


def _parse_agent_payload(raw_text: str) -> Optional[Dict[str, Any]]:
    candidate = _extract_json_candidate(raw_text) if raw_text else None
    payloads = _decode_json_objects(candidate) if candidate else []
    if not payloads and raw_text:
        payloads = _decode_json_objects(raw_text)
    if not payloads:
        return None

    merged: Dict[str, Any] = {}
    note_parts: List[str] = []
    for payload in payloads:
        for key, value in payload.items():
            if key == "notes":
                if value:
                    note_parts.append(str(value))
                continue
            merged[key] = value

    if note_parts:
        merged["notes"] = " | ".join(part.strip() for part in note_parts if str(part).strip())

    return merged


def _judge_patch(entry: Dict[str, Any], payload: Dict[str, Any]) -> tuple[bool, List[str]]:
    """Apply lightweight judging to ensure the payload is actionable."""
    issues: List[str] = []
    contract_id = payload.get("contract_id")
    if contract_id and contract_id != entry["contract_id"]:
        issues.append(
            f"contract_id mismatch (expected {entry['contract_id']}, got {contract_id})"
        )

    missing_after_patch = [
        field for field in entry["missing_fields"] if value_missing(payload.get(field))
    ]
    if missing_after_patch:
        issues.append(
            "Still missing: " + ", ".join(sorted(missing_after_patch))
        )

    confidence = payload.get("confidence")
    if confidence and isinstance(confidence, str) and confidence.lower() not in {"high", "medium"}:
        issues.append("Confidence not provided or too low")

    return len(issues) == 0, issues


def _apply_patch(db: ContractVectorDB, entry: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    """Write metadata updates back to ChromaDB."""
    updates: Dict[str, Any] = {}
    for field in entry["missing_fields"]:
        candidate = payload.get(field)
        if value_missing(candidate):
            continue
        updates[field] = _coerce_metadata_value(candidate)
    if not updates:
        return {}
    db.collection.update(ids=[entry["contract_id"]], metadatas=[updates])
    return updates


def _apply_direct_updates(db: ContractVectorDB, contract_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
    sanitized: Dict[str, Any] = {}
    for field, value in updates.items():
        if value_missing(value):
            continue
        sanitized[field] = _coerce_metadata_value(value)
    if not sanitized:
        return {}
    db.collection.update(ids=[contract_id], metadatas=[sanitized])
    return sanitized


def _display_values(values: Dict[str, Any]) -> Dict[str, str]:
    return {field: coerce_text(value) for field, value in values.items()}


def _write_log(log_path: Path, record: Dict[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ADK to fill metadata gaps automatically")
    parser.add_argument("--top-prompts", type=int, default=5, help="How many gap prompts to attempt")
    parser.add_argument("--snippet-chars", type=int, default=900, help="Snippet length for prompts")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override cloud Gemini model id (only used when not running locally)",
    )
    parser.add_argument(
        "--local-model",
        type=str,
        default=None,
        help="Override local Ollama profile name (implies local execution)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Do not write back to ChromaDB")
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Optional log file path (defaults to metadata_gap_runs/<timestamp>.jsonl)",
    )
    args = parser.parse_args()

    descriptions = {field: desc for field, desc in HIGH_VALUE_FIELDS}
    _, gap_entries, _ = collect_gap_data(args.snippet_chars)

    if not gap_entries:
        print("✅ No metadata gaps detected. Nothing to run.")
        return

    db = ContractVectorDB()
    llm_model = _build_runner_llm(args.model, args.local_model)
    agent = MetadataGapAgent(llm_model=llm_model)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    log_path = Path(args.log_file) if args.log_file else Path("metadata_gap_runs") / f"run-{timestamp}.jsonl"

    attempted = gap_entries[: args.top_prompts]
    successes = 0
    for entry in attempted:
        inherited_updates = cast(Dict[str, Any], entry.get("inherited_metadata") or {})
        inherited_source = entry.get("inherited_from_contract")
        if inherited_updates:
            applied = _apply_direct_updates(db, entry["contract_id"], inherited_updates)
            if applied:
                source_label = f"{inherited_source}" if inherited_source else "related master agreement"
                display_prefills = _display_values({field: inherited_updates[field] for field in applied})
                applied_preview = "; ".join(
                    f"{field}={value}" for field, value in display_prefills.items()
                )
                print(f"ℹ️  Adopted metadata from {source_label}: {applied_preview}")
                entry["missing_fields"] = [field for field in entry.get("missing_fields", []) if field not in applied]
                if not entry.get("missing_fields"):
                    log_record: Dict[str, Any] = {
                        "contract_id": entry["contract_id"],
                        "missing_fields": [],
                        "status": "prefilled",
                        "timestamp": datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"),
                        "prefill_fields": list(applied.keys()),
                        "prefill_source": inherited_source,
                        "prefill_values": display_prefills,
                    }
                    _write_log(log_path, log_record)
                    successes += 1
                    continue

        prompt = build_adk_prompt(entry, descriptions)
        print(f"\n=== Processing {entry['contract_id']} ({len(entry['missing_fields'])} missing fields) ===")
        record: Dict[str, Any] = {
            "contract_id": entry["contract_id"],
            "missing_fields": entry["missing_fields"],
            "prompt_chars": len(prompt),
            "response": "",
            "status": "",
            "timestamp": timestamp,
        }
        try:
            agent_response = agent.run_prompt(prompt)
            record["response"] = agent_response
        except RuntimeError as err:
            record["status"] = "agent_error"
            record["notes"] = str(err)
            _write_log(log_path, record)
            print(f"❌ Agent call failed: {err}")
            continue
        payload = _parse_agent_payload(agent_response)

        if not payload:
            record["status"] = "parse_failed"
            record["notes"] = "Could not parse JSON from agent response"
            _write_log(log_path, record)
            print("❌ ADK response was not valid JSON")
            continue

        is_valid, judge_notes = _judge_patch(entry, payload)
        if not is_valid:
            record["status"] = "rejected"
            record["notes"] = "; ".join(judge_notes)
            record["payload"] = payload
            _write_log(log_path, record)
            print("⚠️ Judge rejected payload:", record["notes"])
            continue

        if args.dry_run:
            record["status"] = "dry_run"
            record["payload"] = payload
            _write_log(log_path, record)
            print("📝 Dry run successful; metadata not written")
            successes += 1
            continue

        applied = _apply_patch(db, entry, payload)
        if not applied:
            record["status"] = "no_updates"
            record["payload"] = payload
            record["notes"] = "Payload validated but no actionable fields"
            _write_log(log_path, record)
            print("⚠️ Nothing to update after validation")
            continue

        record["status"] = "applied"
        record["payload"] = payload
        record["applied_fields"] = applied
        record["applied_fields_display"] = _display_values(applied)
        _write_log(log_path, record)
        applied_preview = ", ".join(
            f"{field}={value}" for field, value in record["applied_fields_display"].items()
        )
        print(f"✅ Updated fields: {applied_preview}")
        successes += 1

    print("\n=== RUN SUMMARY ===")
    print(f"Attempted prompts: {len(attempted)}")
    print(f"Successful patches: {successes}")
    print(f"Log file: {log_path}")


if __name__ == "__main__":
    main()
