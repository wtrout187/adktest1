"""Scan Chroma metadata for high-value gaps and emit ADK-ready prompts."""
from __future__ import annotations

import argparse
from collections import defaultdict
from textwrap import dedent
from typing import Any, Dict, List, Optional, Sized, Tuple, TypedDict, cast

from clm_with_chromadb import ContractVectorDB


class GroupEntry(TypedDict):
    contract_id: str
    metadata: Dict[str, Any]

HIGH_VALUE_FIELDS = [
    ("auto_renewal", "Auto-renewal flag"),
    ("perpetual", "Perpetual indicator"),
    ("perpetual_notice_period", "Perpetual notice requirement"),
    ("renewal_notice_period", "Renewal notice window"),
    ("termination_notice_period", "Termination notice window"),
    ("termination_for_convenience_notice", "Termination for convenience notice"),
    ("payment_terms", "Payment terms"),
    ("gdpr_applicable", "GDPR applicability string"),
    ("subject_to_gdpr", "GDPR boolean flag"),
    ("counterparty_address", "Counterparty mailing address"),
    ("spectralink_address", "Spectralink mailing address"),
    ("governing_law", "Governing law"),
    ("counterparty_name", "Counterparty name"),
    ("primary_party_name", "Primary party name"),
]

PROMPT_META_FIELDS = [
    "contract_type",
    "parties",
    "counterparty_name",
    "primary_party_name",
    "effective_date",
    "termination_date",
    "region",
    "gdpr_reason",
]

MISSING_STRINGS = {"", "unknown", "n/a", "na", "not found", "tbd", "pending"}

MASTER_PARENT_KEYWORDS = (
    "master service agreement",
    "msa",
    "master sales agreement",
    "distributor agreement",
    "oem agreement",
    "partner program agreement",
    "reseller agreement",
    "master agreement",
)

NDA_EXCLUDED_FIELDS = {"payment_terms"}
SKIP_DRAFT_TOKENS = {"draft"}


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _collect_contract_categories(metadata: Dict[str, Any]) -> List[str]:
    contract_text = _normalize_text(metadata.get("contract_type"))
    role_text = _normalize_text(metadata.get("document_role"))
    folder_text = _normalize_text(metadata.get("folder_name"))
    filename_text = _normalize_text(metadata.get("filename"))
    relative_text = _normalize_text(metadata.get("relative_path"))
    combined = " ".join(filter(None, [contract_text, role_text, folder_text, filename_text, relative_text]))

    categories: List[str] = []
    is_master = any(keyword in combined for keyword in MASTER_PARENT_KEYWORDS)
    document_sequence = _normalize_text(metadata.get("document_sequence_label"))
    is_sow = (
        "statement of work" in role_text
        or "statement of work" in contract_text
        or " sow" in f" {filename_text} "
        or document_sequence.startswith("sow")
    )
    is_schedule = "schedule" in role_text or "schedule" in filename_text or "schedule" in folder_text
    is_amendment = (
        "amendment" in role_text
        or "addendum" in role_text
        or "change order" in combined
        or (is_schedule and not is_master)
    )
    nda_triggers = (
        "nda" in combined
        or "non-disclosure" in combined
        or "confidentiality" in combined
    )

    if is_master:
        categories.append("master")
    if is_sow and not is_master:
        categories.append("sow")
    if is_amendment:
        categories.append("amendment")
    if is_schedule and not is_amendment:
        categories.append("schedule")
    if nda_triggers:
        categories.append("nda")
    return categories


def _should_skip_entry(metadata: Dict[str, Any]) -> bool:
    if bool(metadata.get("is_draft")):
        return True
    status_label = _normalize_text(metadata.get("document_status_label"))
    if any(token in status_label for token in SKIP_DRAFT_TOKENS):
        return True
    folder_name = _normalize_text(metadata.get("folder_name"))
    if any(token in folder_name for token in SKIP_DRAFT_TOKENS):
        return True
    parent_ref = _normalize_text(metadata.get("document_parent_reference"))
    if any(token in parent_ref for token in SKIP_DRAFT_TOKENS):
        return True
    return False


def _build_relationship_hint(metadata: Dict[str, Any]) -> Optional[str]:
    hints: List[str] = []
    parent_ref = _normalize_text(metadata.get("document_parent_reference"))
    if parent_ref:
        hints.append(f"Folder lineage points to '{metadata['document_parent_reference']}'.")
    related_hint = metadata.get("related_contract_hint")
    if related_hint:
        hints.append(str(related_hint))
    expected_related = metadata.get("expected_related_documents")
    if expected_related:
        hints.append(f"Expected related documents: {expected_related}.")
    return " ".join(hints) if hints else None


def _apply_contract_specialist_logic(
    metadata: Dict[str, Any], missing_fields: List[str]
) -> tuple[List[str], List[str], Optional[str], List[str]]:
    categories = _collect_contract_categories(metadata)
    suppressed_fields: List[str] = []
    guidance_parts: List[str] = []

    if "nda" in categories:
        suppressed_fields.extend([field for field in missing_fields if field in NDA_EXCLUDED_FIELDS])
        guidance_parts.append(
            "Mutual NDA documents typically cover confidentiality obligations only; payment terms and similar commercial details are handled in separate commercial agreements."
        )

    if "sow" in categories:
        hint = _build_relationship_hint(metadata)
        guidance = (
            "Statement of Work (SOW) documents spell out project-specific deliverables, fees, milestones, and end dates under a governing master agreement. "
            "Core commercial terms—payment, governing law, indemnities, notice mechanics—stay controlled by the master unless the SOW explicitly restates them. "
            "Termination or expiration dates inside a SOW describe that work order's timeline, not the overall relationship."
        )
        if hint:
            guidance = f"{guidance} {hint}"
        guidance_parts.append(
            f"{guidance} Capture values only if the SOW expressly restates the field; otherwise cite the master agreement (e.g., 'per MSA dated 16 Oct 2015')."
        )

    if "amendment" in categories:
        hint = _build_relationship_hint(metadata)
        guidance = (
            "Amendments/change orders are short instruments that revise specific sections of the governing agreement. "
            "Unless the amendment replaces an entire clause, assume the master agreement still controls legal boilerplate (payment, governing law, notices, addresses)."
        )
        if hint:
            guidance = f"{guidance} {hint}"
        guidance_parts.append(guidance)

    if "schedule" in categories:
        hint = _build_relationship_hint(metadata)
        guidance = (
            "Schedules and exhibits are attachments to the master agreement that provide detailed pricing tables, service descriptions, or localization specifics. "
            "They rarely introduce new legal constructs; instead they reference the master for remedies and definitions."
        )
        if hint:
            guidance = f"{guidance} {hint}"
        guidance_parts.append(
            f"{guidance} Treat them like subordinate documents: pull values only when the schedule clearly overrides the governing contract."
        )

    filtered_missing = [field for field in missing_fields if field not in suppressed_fields]
    guidance_text = " ".join(guidance_parts) if guidance_parts else None
    return filtered_missing, suppressed_fields, guidance_text, categories


def _locate_master_contract(
    contract_id: str,
    metadata: Dict[str, Any],
    categories: List[str],
    group_index: Dict[str, List[GroupEntry]],
) -> Optional[GroupEntry]:
    if not categories or not any(category in {"sow", "amendment", "schedule"} for category in categories):
        return None
    group_id = metadata.get("contract_group_id")
    if not group_id:
        return None
    candidates = group_index.get(group_id, [])
    master_candidate: Optional[GroupEntry] = None
    for candidate in candidates:
        if candidate["contract_id"] == contract_id:
            continue
        candidate_meta = candidate["metadata"]
        candidate_categories = _collect_contract_categories(candidate_meta)
        if "master" in candidate_categories:
            master_candidate = candidate
            break
        candidate_text = " ".join(
            filter(
                None,
                [
                    _normalize_text(candidate_meta.get("contract_type")),
                    _normalize_text(candidate_meta.get("document_role")),
                ],
            )
        )
        if any(keyword in candidate_text for keyword in MASTER_PARENT_KEYWORDS):
            master_candidate = candidate
            break
    return master_candidate


def _inherit_metadata_from_master(
    contract_id: str,
    metadata: Dict[str, Any],
    categories: List[str],
    missing_fields: List[str],
    group_index: Dict[str, List[GroupEntry]],
) -> tuple[Dict[str, Any], Optional[str]]:
    parent = _locate_master_contract(contract_id, metadata, categories, group_index)
    if not parent:
        return {}, None
    inherited: Dict[str, Any] = {}
    parent_metadata = parent["metadata"]
    for field in missing_fields:
        value = parent_metadata.get(field)
        if not value_missing(value):
            inherited[field] = value
    if not inherited:
        return {}, None
    return inherited, parent["contract_id"]


def value_missing(value: Any) -> bool:
    """Return True when a metadata value should be treated as missing."""
    if value is None:
        return True
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return False
    if isinstance(value, str):
        normalized = value.strip().lower()
        return normalized in MISSING_STRINGS
    if isinstance(value, (list, tuple, set, dict)):
        sized_value = cast(Sized, value)
        return len(sized_value) == 0
    return False


def coerce_text(value: Any) -> str:
    if value is None:
        return "Unknown"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    return str(value)


def build_adk_prompt(entry: Dict[str, Any], descriptions: Dict[str, str]) -> str:
    missing_descriptions = ", ".join(
        f"{field} ({descriptions.get(field, field)})" for field in entry["missing_fields"]
    )
    known_lines = [
        f"- {field}: {coerce_text(entry['metadata'].get(field))}"
        for field in PROMPT_META_FIELDS
    ]
    specialist_sections: List[str] = []
    specialist_note = entry.get("contract_specialist_note")
    if specialist_note:
        specialist_sections.append(f"Contracts specialist guidance: {specialist_note}")
    suppressed_fields = cast(List[str], entry.get("suppressed_fields") or [])
    if suppressed_fields:
        suppressed_text = ", ".join(sorted(set(suppressed_fields)))
        specialist_sections.append(
            f"Fields treated as out-of-scope for this document type: {suppressed_text}."
        )
    inherited_metadata = cast(Dict[str, Any], entry.get("inherited_metadata") or {})
    if inherited_metadata:
        inherited_from = entry.get("inherited_from_contract") or "related master agreement"
        inherited_lines = "\n".join(
            f"  • {field}: {coerce_text(value)}" for field, value in inherited_metadata.items()
        )
        specialist_sections.append(
            "Potential inherited values from "
            f"{inherited_from}:\n{inherited_lines}\n  • Verify whether the subordinate document restates or overrides each value before applying."
        )
    specialist_block = ("\n" + "\n".join(specialist_sections)) if specialist_sections else ""
    snippet = entry["excerpt"]
    return dedent(
        f"""
        You are the Google ADK Phase 2 metadata validation agent for Spectralink's CLM system.
        Contract ID: {entry['contract_id']}
        Missing metadata: {missing_descriptions}

        Known context:
        {chr(10).join(known_lines)}

        {specialist_block}

        Contract excerpt (truncated to {len(snippet)} chars):
        <<<
        {snippet}
        >>>

        Instructions:
        1. Use the excerpt (and broader contract text if needed) to determine the missing values.
        2. Return a JSON object with keys for each missing field plus "confidence" and optional "notes".
           Example:
           {{
             "contract_id": "{entry['contract_id']}",
             "auto_renewal": "Yes",
             "renewal_notice_period": "90 days",
             "confidence": "high",
             "notes": "Notice window appears in Section 12"
           }}
          3. Only include fields you can support with text evidence. If uncertain, omit the field and set
              "confidence": "low" with an explanatory note.
          4. When the document is a SOW, amendment, or other child document, cross-check the controlling master
              agreement before copying inherited values. If a field is governed exclusively by the master,
              set "notes" to indicate the dependency (e.g., "Payment terms per MSA dated 1 Jan 2024").
        """
    ).strip()


def collect_gap_data(
    snippet_chars: int,
) -> Tuple[Dict[str, List[str]], List[Dict[str, Any]], Dict[str, List[str]]]:
    db = ContractVectorDB()
    snapshot = db.collection.get(include=["metadatas", "documents"])
    metadatas = [dict(md or {}) for md in snapshot.get("metadatas") or []]
    documents = snapshot.get("documents") or []
    ids = snapshot.get("ids") or []

    summary: Dict[str, List[str]] = defaultdict(list)
    inherited_summary: Dict[str, List[str]] = defaultdict(list)
    gap_entries: List[Dict[str, Any]] = []

    group_index: Dict[str, List[GroupEntry]] = defaultdict(list)
    for idx, contract_id in enumerate(ids):
        metadata = metadatas[idx] if idx < len(metadatas) else {}
        group_id = metadata.get("contract_group_id")
        if group_id:
            group_index[str(group_id)].append(
                GroupEntry(contract_id=str(contract_id), metadata=metadata)
            )

    for idx, contract_id in enumerate(ids):
        metadata = metadatas[idx] if idx < len(metadatas) else {}
        if _should_skip_entry(metadata):
            continue
        raw_missing: List[str] = []
        for field, _ in HIGH_VALUE_FIELDS:
            if value_missing(metadata.get(field)):
                raw_missing.append(field)
        if not raw_missing:
            continue

        filtered_missing, suppressed_fields, guidance_text, categories = _apply_contract_specialist_logic(
            metadata, raw_missing
        )
        inherited_metadata, inherited_from = _inherit_metadata_from_master(
            str(contract_id), metadata, categories, filtered_missing, group_index
        )
        inherited_fields = set(inherited_metadata.keys())
        remaining_missing = [field for field in filtered_missing if field not in inherited_fields]

        if not remaining_missing and not inherited_metadata:
            continue

        for field in remaining_missing:
            summary[field].append(contract_id)
        for field in inherited_fields:
            inherited_summary[field].append(contract_id)

        text = documents[idx] if idx < len(documents) else ""
        snippet = (text[: snippet_chars] + "…") if len(text) > snippet_chars else text
        entry: Dict[str, Any] = {
            "contract_id": contract_id,
            "missing_fields": remaining_missing,
            "metadata": metadata,
            "excerpt": snippet,
            "suppressed_fields": suppressed_fields,
            "contract_specialist_note": guidance_text,
            "contract_categories": categories,
        }
        if inherited_metadata:
            entry["inherited_metadata"] = inherited_metadata
            entry["inherited_from_contract"] = inherited_from
            entry["suppressed_inherited_fields"] = sorted(inherited_fields)

        gap_entries.append(entry)

    gap_entries.sort(key=lambda entry: len(entry["missing_fields"]), reverse=True)
    return summary, gap_entries, inherited_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan metadata and surface gap prompts")
    parser.add_argument("--top-prompts", type=int, default=3, help="How many ADK prompts to emit")
    parser.add_argument(
        "--snippet-chars",
        type=int,
        default=900,
        help="Maximum characters to include in each contract excerpt",
    )
    args = parser.parse_args()

    summary, gap_entries, inherited_summary = collect_gap_data(args.snippet_chars)
    descriptions = {field: desc for field, desc in HIGH_VALUE_FIELDS}

    print("\n=== HIGH-VALUE METADATA GAPS ===")
    if not gap_entries:
        print("All monitored fields are populated. 🚀")
        return

    for field, desc in HIGH_VALUE_FIELDS:
        missing_list = summary.get(field, [])
        if missing_list:
            print(f"- {field} ({desc}): {len(missing_list)} contracts missing")

    print("\n=== ADK PROMPTS (copy/paste) ===")
    for entry in gap_entries[: args.top_prompts]:
        prompt = build_adk_prompt(entry, descriptions)
        print("\n" + "-" * 70)
        print(prompt)
        print("-" * 70)

    if inherited_summary:
        print("\n=== SUPPRESSED VIA MASTER AGREEMENTS ===")
        for field, contracts in inherited_summary.items():
            desc = descriptions.get(field, field)
            print(f"- {field} ({desc}): inherited for {len(contracts)} subordinate docs")


if __name__ == "__main__":
    main()
