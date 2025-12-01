"""
Enterprise Contract Lifecycle Management (CLM) System
=====================================================

A production-ready CLM system built with Google ADK featuring:
- Memory & session state management
- Safety guardrails with callbacks
- Multi-agent architecture
- SharePoint integration
- Document versioning & relationship tracking
- Evaluation framework
- Local/cloud processing options

Author: ADKTest Project
License: MIT
"""

import os
import re
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

# Load environment variables
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / "my_agent" / ".env")

# ADK imports
from google.adk.agents import Agent
from google.adk.models import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.tool_context import ToolContext
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.tools.base_tool import BaseTool
from google.genai import types

from contract_patterns import (
    detect_pii,
    extract_defined_terms,
    extract_renewal_terms,
    find_contract_sections,
)


# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

APP_NAME = "clm_enterprise_system"
DEFAULT_USER_ID = "clm_user_001"

# Session state keys
STATE_KEYS = {
    "user_department": "user_department",  # Legal, Sales, Procurement, etc.
    "sensitivity_level": "sensitivity_level",  # public, internal, confidential, restricted
    "last_contract_type": "last_contract_type",  # MSA, NDA, SOW, etc.
    "analysis_history": "analysis_history",  # List of analyzed contracts
    "user_preferences": "user_preferences",  # Dict of user-specific settings
    "blocked_keywords": "blocked_keywords",  # Security - blocked terms
}

# Contract types
CONTRACT_TYPES = [
    "Master Service Agreement (MSA)",
    "Non-Disclosure Agreement (NDA)",
    "OEM Agreement",
    "Distributor Agreement",
    "Statement of Work (SOW)",
    "License Agreement",
    "Amendment",
    "Work Order"
]

# ============================================================================
# CONTRACT ANALYSIS TOOLS
# ============================================================================

def extract_contract_metadata(contract_text: str, tool_context: ToolContext) -> str:
    """
    Extract comprehensive metadata from contract text including parties, dates, 
    terms, and defined terms. Uses session state to track analysis history.
    
    Args:
        contract_text: Full text of the contract
        tool_context: ADK tool context with session state access
    
    Returns:
        JSON string with extracted metadata
    """
    print(f"--- Tool: extract_contract_metadata called ---")
    print(f"--- Tool: Reading user department from state: {tool_context.state.get('user_department', 'Not Set')} ---")
    
    metadata: Dict[str, Any] = {
        "analysis_timestamp": datetime.now().isoformat(),
        "analyzer_department": tool_context.state.get("user_department", "Unknown"),
        "parties": [],
        "effective_date": None,
        "termination_date": None,
        "contract_type": None,
        "defined_terms": [],
        "key_sections": {},
        "renewal_info": {}
    }
    
    # Extract parties
    party_patterns = [
        r'between\s+(.*?)\s+and\s+(.*?)(?:\s+dated|\s+effective|\.|$)',
        r'This\s+Agreement.*?by\s+and\s+between\s+(.*?)\s+and\s+(.*?)(?:\.|,)'
    ]
    for pattern in party_patterns:
        match = re.search(pattern, contract_text, re.IGNORECASE | re.DOTALL)
        if match:
            metadata["parties"] = [match.group(1).strip(), match.group(2).strip()]
            break
    
    # Extract dates
    date_pattern = r'(?:effective\s+date|dated|executed\s+on).*?(\w+\s+\d{1,2},?\s+\d{4})'
    date_match = re.search(date_pattern, contract_text, re.IGNORECASE)
    if date_match:
        metadata["effective_date"] = date_match.group(1)
    
    # Identify contract type
    for contract_type in CONTRACT_TYPES:
        if contract_type.lower() in contract_text.lower():
            metadata["contract_type"] = contract_type
            tool_context.state[STATE_KEYS["last_contract_type"]] = contract_type
            print(f"--- Tool: Updated state 'last_contract_type': {contract_type} ---")
            break
    
    # Extract defined terms
    metadata["defined_terms"] = extract_defined_terms(contract_text)
    
    # Find key sections
    metadata["key_sections"] = find_contract_sections(contract_text)
    
    # Extract renewal information
    metadata["renewal_info"] = extract_renewal_terms(contract_text)
    
    # Update analysis history in state
    history = tool_context.state.get(STATE_KEYS["analysis_history"], [])
    history.append({
        "timestamp": metadata["analysis_timestamp"],
        "contract_type": metadata["contract_type"],
        "parties": metadata["parties"]
    })
    tool_context.state[STATE_KEYS["analysis_history"]] = history[-10:]  # Keep last 10
    print(f"--- Tool: Updated analysis history (count: {len(history)}) ---")
    
    return json.dumps(metadata, indent=2)


def analyze_contract_risks(contract_text: str, tool_context: ToolContext) -> str:
    """
    Analyze potential risks in the contract with sensitivity-based filtering.
    Risk analysis adapts based on user's sensitivity clearance level.
    
    Args:
        contract_text: Full text of the contract
        tool_context: ADK tool context with session state
    
    Returns:
        JSON string with risk analysis
    """
    print(f"--- Tool: analyze_contract_risks called ---")
    
    sensitivity_level = tool_context.state.get(STATE_KEYS["sensitivity_level"], "internal")
    print(f"--- Tool: User sensitivity level: {sensitivity_level} ---")
    
    high_risk: List[Dict[str, Any]] = []
    medium_risk: List[Dict[str, Any]] = []
    low_risk: List[Dict[str, Any]] = []
    recommendations: List[str] = []

    risks: Dict[str, Any] = {
        "analysis_timestamp": datetime.now().isoformat(),
        "sensitivity_level": sensitivity_level,
        "high_risk": high_risk,
        "medium_risk": medium_risk,
        "low_risk": low_risk,
        "recommendations": recommendations
    }
    
    # High-risk patterns
    if re.search(r'unlimited.*liability', contract_text, re.IGNORECASE):
        high_risk.append({
            "risk": "Unlimited Liability Clause",
            "description": "Contract contains unlimited liability provisions",
            "severity": "HIGH"
        })
    
    if re.search(r'non-compete', contract_text, re.IGNORECASE):
        high_risk.append({
            "risk": "Non-Compete Clause",
            "description": "Restrictive covenant that may limit business operations",
            "severity": "HIGH"
        })
    
    # Medium-risk patterns
    renewal_info = extract_renewal_terms(contract_text)
    if renewal_info["has_auto_renewal"]:
        medium_risk.append({
            "risk": "Auto-Renewal Clause",
            "description": f"Contract auto-renews with {renewal_info.get('notice_period_days', 'unspecified')} days notice required",
            "severity": "MEDIUM"
        })
    
    if re.search(r'indemnif', contract_text, re.IGNORECASE):
        medium_risk.append({
            "risk": "Indemnification Provisions",
            "description": "Contract contains indemnification clauses requiring legal review",
            "severity": "MEDIUM"
        })
    
    # Low-risk observations
    if not re.search(r'confidential', contract_text, re.IGNORECASE):
        low_risk.append({
            "risk": "No Confidentiality Clause",
            "description": "Contract lacks explicit confidentiality provisions",
            "severity": "LOW"
        })
    
    # Recommendations based on sensitivity
    if sensitivity_level in ["confidential", "restricted"]:
        recommendations.append("Recommend executive review for high-sensitivity contract")
        recommendations.append("Ensure data classification labels are properly applied")
    else:
        recommendations.append("Standard risk review process applies")
    
    return json.dumps(risks, indent=2)


def generate_contract_summary(contract_text: str, tool_context: ToolContext) -> str:
    """
    Generate executive summary of the contract with personalization
    based on user department.
    
    Args:
        contract_text: Full text of the contract
        tool_context: ADK tool context
    
    Returns:
        JSON string with summary
    """
    print(f"--- Tool: generate_contract_summary called ---")
    
    department = tool_context.state.get(STATE_KEYS["user_department"], "General")
    
    summary: Dict[str, Any] = {
        "generated_for_department": department,
        "timestamp": datetime.now().isoformat(),
        "contract_type": None,
        "key_parties": [],
        "term_length": None,
        "key_obligations": [],
        "critical_dates": [],
        "executive_summary": ""
    }
    
    # Extract basic info
    contract_type = tool_context.state.get(STATE_KEYS["last_contract_type"], "Unknown")
    summary["contract_type"] = contract_type
    
    # Build department-specific summary
    if department == "Legal":
        summary["executive_summary"] = f"Legal Review Summary: {contract_type} contract requires attention to liability provisions, indemnification clauses, and governing law."
    elif department == "Sales":
        summary["executive_summary"] = f"Sales Overview: {contract_type} focusing on payment terms, delivery obligations, and revenue recognition impacts."
    elif department == "Procurement":
        summary["executive_summary"] = f"Procurement Analysis: {contract_type} evaluated for supplier obligations, pricing terms, and compliance requirements."
    else:
        summary["executive_summary"] = f"General Summary: {contract_type} overview for cross-functional review."
    
    # Extract term length
    term_match = re.search(r'(?:initial\s+term|term\s+of).*?(\d+)\s+(year|month)', contract_text, re.IGNORECASE)
    if term_match:
        summary["term_length"] = f"{term_match.group(1)} {term_match.group(2)}(s)"
    
    return json.dumps(summary, indent=2)


def map_contract_relationships(contract_info: str, tool_context: ToolContext) -> str:
    """
    Map relationships between contracts (MSA -> SOW -> Amendments).
    Tracks document hierarchy and version chains.
    
    Args:
        contract_info: Information about the contract to map
        tool_context: ADK tool context
    
    Returns:
        JSON string with relationship mapping
    """
    print(f"--- Tool: map_contract_relationships called ---")
    
    parent_contracts: List[str] = []
    child_contracts: List[str] = []
    amendments: List[str] = []
    related_documents: List[str] = []

    relationships: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "current_contract_type": tool_context.state.get(STATE_KEYS["last_contract_type"]),
        "parent_contracts": parent_contracts,
        "child_contracts": child_contracts,
        "amendments": amendments,
        "related_documents": related_documents,
        "document_hierarchy": ""
    }
    
    # Analyze the analysis history for relationships
    history = tool_context.state.get(STATE_KEYS["analysis_history"], [])
    
    # Build hierarchy based on contract types
    current_type = relationships["current_contract_type"]
    
    if current_type == "Statement of Work (SOW)":
        relationships["document_hierarchy"] = "MSA (Parent) -> SOW (Current)"
        parent_contracts.append("Master Service Agreement (MSA)")
        related_documents.append("Review parent MSA for governing terms")
    
    elif current_type == "Amendment":
        relationships["document_hierarchy"] = "Original Contract -> Amendment (Current)"
        parent_contracts.append("Original Contract")
        related_documents.append("Identify base contract for amendment tracking")
    
    elif current_type == "Master Service Agreement (MSA)":
        relationships["document_hierarchy"] = "MSA (Current) -> SOWs (Children)"
        child_contracts.append("Statements of Work (SOWs)")
        related_documents.append("Track all SOWs executed under this MSA")
    
    # Add info from recent history
    if len(history) > 1:
        related_documents.append(f"Recently analyzed {len(history)} contracts in this session")
    
    return json.dumps(relationships, indent=2)


# ============================================================================
# SAFETY GUARDRAILS - CALLBACKS
# ============================================================================

def input_safety_guardrail(
    callback_context: CallbackContext, 
    llm_request: LlmRequest
) -> Optional[LlmResponse]:
    """
    Before-model callback: Screen inputs for safety concerns including:
    - Blocked keywords from session state
    - PII detection
    - Prompt injection attempts
    
    Returns LlmResponse to block, or None to allow.
    """
    agent_name = callback_context.agent_name
    print(f"--- Callback: input_safety_guardrail running for agent: {agent_name} ---")
    
    # Extract last user message
    last_user_message_text = ""
    if llm_request.contents:
        for content in reversed(llm_request.contents):
            if content.role == 'user' and content.parts:
                if content.parts[0].text:
                    last_user_message_text = content.parts[0].text
                    break
    
    print(f"--- Callback: Inspecting input (first 100 chars): '{last_user_message_text[:100]}...' ---")
    
    # Check for PII
    pii_found = detect_pii(last_user_message_text)
    if pii_found:
        print(f"--- Callback: BLOCKED - PII detected: {pii_found} ---")
        callback_context.state["security_alert"] = f"PII detected: {pii_found}"
        
        return LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part(text=f"⚠️ Security Alert: I cannot process this request because it contains potentially sensitive information ({', '.join(pii_found)}). Please remove personal identifiable information and try again.")],
            )
        )
    
    # Check blocked keywords from state
    blocked_keywords = callback_context.state.get(STATE_KEYS["blocked_keywords"], [])
    for keyword in blocked_keywords:
        if keyword.lower() in last_user_message_text.lower():
            print(f"--- Callback: BLOCKED - Keyword '{keyword}' found ---")
            callback_context.state["security_alert"] = f"Blocked keyword: {keyword}"
            
            return LlmResponse(
                content=types.Content(
                    role="model",
                    parts=[types.Part(text=f"🚫 I cannot process this request because it contains a blocked term. Please rephrase your query without restricted content.")],
                )
            )
    
    # Check for prompt injection attempts
    injection_patterns = [
        r'ignore\s+previous\s+instructions',
        r'disregard\s+all\s+prior',
        r'forget\s+your\s+instructions',
        r'system\s*:\s*you\s+are',
    ]
    
    for pattern in injection_patterns:
        if re.search(pattern, last_user_message_text, re.IGNORECASE):
            print(f"--- Callback: BLOCKED - Potential prompt injection detected ---")
            callback_context.state["security_alert"] = "Prompt injection attempt"
            
            return LlmResponse(
                content=types.Content(
                    role="model",
                    parts=[types.Part(text="🛡️ Security Notice: This request cannot be processed. Please provide a straightforward contract analysis query.")],
                )
            )
    
    # All checks passed
    print(f"--- Callback: Input safety checks passed for {agent_name} ---")
    return None


def tool_safety_guardrail(
    tool: BaseTool, 
    args: Dict[str, Any], 
    tool_context: ToolContext
) -> Optional[Dict[str, Any]]:
    """
    Before-tool callback: Validate tool arguments and enforce policies.
    
    Returns dictionary to override tool execution, or None to allow.
    """
    tool_name = tool.name
    agent_name = tool_context.agent_name
    print(f"--- Callback: tool_safety_guardrail running for tool '{tool_name}' in agent '{agent_name}' ---")
    print(f"--- Callback: Inspecting args: {args} ---")
    
    # Example: Block analysis of contracts from restricted parties
    if tool_name == "analyze_contract_risks":
        contract_text = args.get("contract_text", "")
        
        # Check if contract involves restricted parties
        restricted_parties = ["Acme Restricted Corp", "Blocked Vendor LLC"]
        for party in restricted_parties:
            if party.lower() in contract_text.lower():
                print(f"--- Callback: BLOCKED - Restricted party '{party}' detected ---")
                tool_context.state["security_alert"] = f"Restricted party: {party}"
                
                return {
                    "analysis_timestamp": datetime.now().isoformat(),
                    "error": "BLOCKED",
                    "message": f"⛔ Risk analysis blocked: Contract involves restricted party '{party}'. Escalate to Legal Compliance team."
                }
    
    # Check sensitivity level for certain tools
    sensitivity_level = tool_context.state.get(STATE_KEYS["sensitivity_level"], "internal")
    
    if tool_name == "extract_contract_metadata" and sensitivity_level == "public":
        print(f"--- Callback: WARNING - Public user attempting metadata extraction ---")
        # Allow but log warning (could block depending on policy)
        tool_context.state["audit_log"] = tool_context.state.get("audit_log", [])
        tool_context.state["audit_log"].append({
            "timestamp": datetime.now().isoformat(),
            "action": "metadata_extraction",
            "user_level": "public",
            "warning": "Public user extracted contract metadata"
        })
    
    print(f"--- Callback: Tool safety checks passed for '{tool_name}' ---")
    return None


# ============================================================================
# AGENT SETUP
# ============================================================================

async def create_clm_system():
    """
    Create the CLM enterprise agent system with all features.
    """
    print("🏗️  Building Enterprise CLM System...")
    print("="*70)
    
    # Create the CLM manager agent with safety callbacks
    clm_manager = Agent(
        name="clm_enterprise_manager",
        model=Gemini(model="gemini-2.5-flash"),
        description="Enterprise CLM manager: Extracts metadata, analyzes risks, generates summaries, and maps contract relationships with security guardrails.",
        instruction="""You are an Enterprise Contract Lifecycle Management assistant.

Your capabilities:
1. Extract comprehensive metadata from contracts (parties, dates, terms, defined terms)
2. Analyze risks with severity ratings and compliance recommendations
3. Generate executive summaries tailored to user departments
4. Map contract relationships (MSA -> SOW -> Amendments)

You have access to session state tracking:
- User department (Legal, Sales, Procurement, etc.)
- Sensitivity clearance level
- Analysis history

CRITICAL SECURITY GUIDELINES:
- Never process contracts containing PII without explicit authorization
- Respect sensitivity levels - escalate high-sensitivity contracts to appropriate teams
- Log all analysis for audit compliance
- Flag any unusual or suspicious content immediately

When analyzing contracts:
1. Always extract metadata first to understand the contract type
2. Perform risk analysis appropriate to user's department
3. Generate summaries with department-specific insights
4. Map relationships to parent/child contracts if applicable

Maintain professional tone and be explicit about limitations or areas requiring human expert review.""",
        tools=[
            extract_contract_metadata,
            analyze_contract_risks,
            generate_contract_summary,
            map_contract_relationships
        ],
        before_model_callback=input_safety_guardrail,
        before_tool_callback=tool_safety_guardrail,
        output_key="last_analysis_result"  # Auto-save final responses to state
    )
    
    print("✅ CLM Enterprise Manager created with safety guardrails!")
    return clm_manager


async def setup_session_service():
    """
    Create and configure session service with initial state.
    """
    print("\n📁 Setting up session management...")
    
    session_service = InMemorySessionService()
    
    # Create initial session with default state
    initial_state: Dict[str, Any] = {
        STATE_KEYS["user_department"]: "Legal",  # Default department
        STATE_KEYS["sensitivity_level"]: "internal",  # Default security level
        STATE_KEYS["last_contract_type"]: None,
        STATE_KEYS["analysis_history"]: [],
        STATE_KEYS["user_preferences"]: {
            "summarization_style": "detailed",  # or "brief"
            "risk_threshold": "medium",  # Show medium and high risks
            "include_recommendations": True
        },
        STATE_KEYS["blocked_keywords"]: ["BLOCK", "RESTRICTED_TERM"]  # Example blocked terms
    }
    
    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=DEFAULT_USER_ID,
        session_id="session_001",
        state=initial_state
    )
    
    print(f"✅ Session created: App='{APP_NAME}', User='{DEFAULT_USER_ID}'")
    print(f"   Department: {initial_state[STATE_KEYS['user_department']]}")
    print(f"   Sensitivity: {initial_state[STATE_KEYS['sensitivity_level']]}")
    
    return session_service, DEFAULT_USER_ID, "session_001"


# ============================================================================
# INTERACTION FUNCTIONS
# ============================================================================

async def call_agent_async(
    query: str, 
    runner: Runner, 
    user_id: str, 
    session_id: str
):
    """
    Send a query to the agent and process the response.
    """
    print(f"\n{'='*70}")
    print(f"👤 User Query: {query}")
    print(f"{'='*70}")
    
    content = types.Content(role='user', parts=[types.Part(text=query)])
    final_response_text = "No response generated."
    
    async for event in runner.run_async(
        user_id=user_id, 
        session_id=session_id, 
        new_message=content
    ):
        if event.is_final_response():
            if event.content and event.content.parts:
                final_response_text = event.content.parts[0].text
            elif event.actions and event.actions.escalate:
                final_response_text = f"⚠️ Agent escalated: {event.error_message or 'Unknown error'}"
            break
    
    print(f"\n🤖 CLM Response:\n{final_response_text}")
    print(f"{'='*70}\n")


async def demo_mode(runner: Runner, user_id: str, session_id: str):
    """
    Run automated demo scenarios showing all features.
    """
    print("\n" + "="*70)
    print("🎬 DEMO MODE: Enterprise CLM System Showcase")
    print("="*70)
    
    # Sample contract text (Spectralink distributor agreement)
    sample_contract = """
DISTRIBUTOR AGREEMENT

This Distributor Agreement ("Agreement") is entered into as of January 15, 2020 (the "Effective Date") 
by and between Spectralink Corporation, a Delaware corporation with offices at 2560 55th Street, 
Boulder, Colorado 80301 ("Spectralink") and Tel-E Connect Systems Ltd., a company organized under 
the laws of Canada with offices at 123 Main Street, Toronto, ON ("Distributor").

1. TERM AND TERMINATION

1.1 Initial Term: This Agreement shall commence on the Effective Date and continue for an initial 
term of two (2) years (the "Initial Term").

1.2 Renewal: Upon expiration of the Initial Term, this Agreement shall automatically renew for 
successive one (1) year terms (each a "Renewal Term") unless either party provides written notice 
of non-renewal at least sixty (60) days prior to the expiration of the then-current term.

1.3 Termination for Cause: Either party may terminate this Agreement for cause upon thirty (30) 
days written notice if the other party materially breaches this Agreement and fails to cure such 
breach within the notice period.

2. DEFINED TERMS

For purposes of this Agreement:
- "Products" means the wireless communication devices and systems manufactured by Spectralink
- "Territory" means Canada
- "Confidential Information" means any non-public information designated as confidential

3. DISTRIBUTOR OBLIGATIONS

Distributor shall use commercially reasonable efforts to promote and distribute the Products 
within the Territory. Distributor shall maintain adequate inventory levels and provide 
customer support services.

IN WITNESS WHEREOF, the parties have executed this Agreement as of the Effective Date.

SPECTRALINK CORPORATION          TEL-E CONNECT SYSTEMS LTD.

By: ___________________          By: ___________________
Name: John Smith                 Name: Jane Doe
Title: VP Sales                  Title: President
Date: January 15, 2020           Date: January 15, 2020
"""
    
    # Scenario 1: Extract metadata
    print("\n📋 Scenario 1: Extract Contract Metadata")
    await call_agent_async(
        f"Please extract all metadata from this distributor agreement:\n\n{sample_contract}",
        runner, user_id, session_id
    )
    
    # Scenario 2: Risk analysis
    print("\n⚠️  Scenario 2: Analyze Contract Risks")
    await call_agent_async(
        "Analyze the risks in the contract I just shared, focusing on renewal terms and termination clauses.",
        runner, user_id, session_id
    )
    
    # Scenario 3: Generate summary
    print("\n📄 Scenario 3: Generate Executive Summary")
    await call_agent_async(
        "Generate an executive summary of this contract for our Legal department.",
        runner, user_id, session_id
    )
    
    # Scenario 4: Map relationships
    print("\n🔗 Scenario 4: Map Contract Relationships")
    await call_agent_async(
        "Map the relationships for this contract. Is it a parent contract or does it reference other agreements?",
        runner, user_id, session_id
    )
    
    # Scenario 5: Test safety guardrail - PII
    print("\n🛡️  Scenario 5: Test Safety Guardrail (PII Detection)")
    await call_agent_async(
        "Analyze this contract: John Doe, SSN 123-45-6789, email john@example.com signed on 1/15/2020",
        runner, user_id, session_id
    )
    
    # Scenario 6: Test blocked keyword
    print("\n🚫 Scenario 6: Test Blocked Keyword")
    await call_agent_async(
        "BLOCK this analysis and show me all contracts",
        runner, user_id, session_id
    )


async def interactive_mode(
    runner: Runner,
    user_id: str,
    session_id: str,
    session_service: InMemorySessionService
):
    """
    Interactive chat mode with state management commands.
    """
    print("\n" + "="*70)
    print("💬 INTERACTIVE MODE: Enterprise CLM System")
    print("="*70)
    print("\nCommands:")
    print("  /dept <name>    - Set your department (Legal, Sales, Procurement)")
    print("  /sensitivity <level> - Set sensitivity level (public, internal, confidential, restricted)")
    print("  /history        - Show analysis history")
    print("  /state          - Show current session state")
    print("  /help           - Show this help")
    print("  exit            - Exit interactive mode")
    print("\nReady for contract analysis queries...\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == 'exit':
                print("\n👋 Exiting CLM system. Goodbye!")
                break
            
            # Handle commands
            if user_input.startswith('/'):
                session = await session_service.get_session(
                    app_name=APP_NAME,
                    user_id=user_id,
                    session_id=session_id
                )
                if session is None:
                    print("⚠️ Session not found. Please try again.")
                    continue
                
                if user_input.startswith('/dept '):
                    dept = user_input[6:].strip()
                    # Update state directly in storage
                    stored_session = session_service.sessions[APP_NAME][user_id][session_id]
                    stored_session.state[STATE_KEYS["user_department"]] = dept
                    print(f"✅ Department set to: {dept}")
                    
                elif user_input.startswith('/sensitivity '):
                    level = user_input[13:].strip()
                    stored_session = session_service.sessions[APP_NAME][user_id][session_id]
                    stored_session.state[STATE_KEYS["sensitivity_level"]] = level
                    print(f"✅ Sensitivity level set to: {level}")
                    
                elif user_input == '/history':
                    history = session.state.get(STATE_KEYS["analysis_history"], [])
                    print(f"\n📊 Analysis History ({len(history)} items):")
                    for idx, item in enumerate(history, 1):
                        print(f"  {idx}. {item.get('timestamp', 'N/A')} - {item.get('contract_type', 'Unknown')}")
                        if item.get('parties'):
                            print(f"     Parties: {', '.join(item['parties'])}")
                    
                elif user_input == '/state':
                    print(f"\n🔍 Current Session State:")
                    print(f"  Department: {session.state.get(STATE_KEYS['user_department'], 'Not Set')}")
                    print(f"  Sensitivity: {session.state.get(STATE_KEYS['sensitivity_level'], 'Not Set')}")
                    print(f"  Last Contract Type: {session.state.get(STATE_KEYS['last_contract_type'], 'None')}")
                    print(f"  Analysis Count: {len(session.state.get(STATE_KEYS['analysis_history'], []))}")
                    
                elif user_input == '/help':
                    print("\n📖 Help:")
                    print("  Ask questions about contract analysis, metadata extraction, risk assessment")
                    print("  Use commands to configure your session preferences")
                    print("  The system remembers your analysis history and preferences")
                
                continue
            
            # Regular query - send to agent
            await call_agent_async(user_input, runner, user_id, session_id)
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Exiting...")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

async def main():
    """
    Main entry point for the CLM enterprise system.
    """
    print("\n" + "="*70)
    print("🏢 ENTERPRISE CONTRACT LIFECYCLE MANAGEMENT SYSTEM")
    print("   Powered by Google ADK with Advanced Features")
    print("="*70)
    
    # Setup
    session_service, user_id, session_id = await setup_session_service()
    clm_agent = await create_clm_system()
    
    # Create runner
    runner = Runner(
        agent=clm_agent,
        app_name=APP_NAME,
        session_service=session_service
    )
    print(f"\n✅ Runner created for agent '{clm_agent.name}'")
    
    # Mode selection
    print("\n" + "="*70)
    print("Choose mode:")
    print("1. Run demo scenarios (automated)")
    print("2. Interactive mode (chat with CLM)")
    print("="*70)
    
    mode = input("\nEnter 1 or 2: ").strip()
    
    if mode == "1":
        await demo_mode(runner, user_id, session_id)
    elif mode == "2":
        await interactive_mode(runner, user_id, session_id, session_service)
    else:
        print("Invalid choice. Running demo mode...")
        await demo_mode(runner, user_id, session_id)
    
    # Final state inspection
    print("\n" + "="*70)
    print("📊 Final Session State Summary")
    print("="*70)
    
    final_session = await session_service.get_session(
        app_name=APP_NAME,
        user_id=user_id,
        session_id=session_id
    )
    
    if final_session:
        print(f"Department: {final_session.state.get(STATE_KEYS['user_department'], 'Not Set')}")
        print(f"Sensitivity: {final_session.state.get(STATE_KEYS['sensitivity_level'], 'Not Set')}")
        print(f"Last Contract: {final_session.state.get(STATE_KEYS['last_contract_type'], 'None')}")
        print(f"Analyses Performed: {len(final_session.state.get(STATE_KEYS['analysis_history'], []))}")
        print(f"Last Result: {final_session.state.get('last_analysis_result', 'None')[:100]}...")
        
        if final_session.state.get("security_alert"):
            print(f"\n⚠️  Security Alerts: {final_session.state['security_alert']}")
    
    print("\n✅ CLM System session complete!")


if __name__ == "__main__":
    asyncio.run(main())
