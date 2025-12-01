"""
Contract Lifecycle Management (CLM) Multi-Agent System - FIXED VERSION
=======================================================================

This demonstrates a hierarchical multi-agent architecture for contract analysis.

ARCHITECTURE:
- Manager Agent: Routes tasks and coordinates specialists
- Simple Tool Functions: Direct contract processing (no nested loops)
- Pattern Matching: Uses regex to extract defined terms like ("Term"), ("Effective Date")

KEY IMPROVEMENTS:
- Fixed function calling compatibility
- Added contract pattern recognition for preamble/term sections
- Switched to gemini-1.5-flash (better quota)
- Proper async handling
"""

import asyncio
import re
from pathlib import Path
from typing import List
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.genai import types

from contract_patterns import (
    defined_terms_to_dict,
    extract_defined_terms,
    extract_renewal_terms,
    extract_section_blocks,
)

# Load environment variables
env_path = Path(__file__).parent / "my_agent" / ".env"
load_dotenv(env_path)

print("🏗️  Building Contract Lifecycle Management System...")
print("="*70)

# =============================================================================
# RETRY CONFIGURATION
# =============================================================================
retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504]
)

# =============================================================================
# TOOL FUNCTIONS - These are callable by the manager agent
# =============================================================================

def extract_contract_metadata(contract_text: str) -> str:
    """
    Extract structured metadata from a contract.
    
    Focuses on:
    - Contract type (MSA, NDA, OEM, Distributor, SOW, etc.)
    - Parties and short names
    - Effective date
    - Term and renewal info
    - Key financial terms
    - Department ownership
    """
    result = ["📋 CONTRACT METADATA EXTRACTION\n" + "="*50]
    
    # Extract defined terms
    defined_term_list = extract_defined_terms(contract_text)
    defined_terms = defined_terms_to_dict(defined_term_list)
    if defined_term_list:
        result.append("\n🔖 Defined Terms:")
        for item in defined_term_list:
            result.append(f"  • {item['term']}: {item['definition']}")
    
    # Find key sections
    sections = extract_section_blocks(contract_text)
    
    # Extract from preamble
    if 'preamble' in sections:
        result.append("\n📄 From Preamble:")
        preamble = sections['preamble']
        
        # Contract type
        for contract_type in ['Master Service Agreement', 'Distributor Agreement', 'OEM Agreement', 
                              'NDA', 'SOW', 'Statement of Work', 'License Agreement']:
            if contract_type.lower() in preamble.lower():
                result.append(f"  • Contract Type: {contract_type}")
                break
        
        # Effective date
        if 'Effective Date' in defined_terms:
            result.append(f"  • Effective Date: {defined_terms['Effective Date']}")
    
    # Extract term info
    if 'term_termination' in sections:
        result.append("\n⏰ Term & Renewal:")
        renewal_info = extract_renewal_terms(sections['term_termination'])
        
        if 'Initial Term' in defined_terms:
            result.append(f"  • Initial Term: {defined_terms['Initial Term']}")
        if 'Renewal Term' in defined_terms:
            result.append(f"  • Renewal Term: {defined_terms['Renewal Term']}")
        
        if renewal_info.get("has_auto_renewal"):
            clause = renewal_info.get("renewal_clause") or renewal_info.get("renewal_term_length", "Renewal clause detected")
            result.append(f"  • Auto-Renewal: Yes ({clause})")
        notice_days = renewal_info.get("notice_period_days")
        if notice_days:
            result.append(f"  • Notice Period: {notice_days} days")
        if renewal_info.get("written_notice_required"):
            result.append("  • Written Notice: Required")
    
    # Extract signature info
    if 'signatures' in sections:
        result.append("\n✍️  Execution:")
        sig_section = sections['signatures']
        dates = re.findall(r'DATE:\s*([^\n]+)', sig_section)
        if dates:
            result.append(f"  • Signed: {', '.join(dates)}")
    
    return "\n".join(result)


def analyze_contract_risks(contract_text: str) -> str:
    """
    Analyze risks in a contract.
    
    Risk categories:
    - Legal (IP, indemnification, liability caps)
    - Financial (payment terms, penalties, minimums)
    - Operational (SLAs, termination rights, exclusivity)
    - Compliance (GDPR, export control, data privacy)
    """
    result = ["⚠️  RISK ANALYSIS\n" + "="*50]
    
    risks_found: List[str] = []
    
    # Check for high-risk terms
    high_risk_patterns = {
        'unlimited liability': r'unlimited\s+liability',
        'no liability cap': r'no\s+(?:cap|limit).*?liability',
        'auto-renewal without notice': r'automatically\s+renew(?!.*notice)',
        'exclusive arrangement': r'exclusiv(?:e|ity)',
        'minimum purchase requirements': r'minimum.*?(?:order|purchase|commitment)',
        'liquidated damages': r'liquidated\s+damages',
        'IP ownership transfer': r'(?:transfer|assign).*?(?:intellectual\s+property|IP)',
    }
    
    for risk_name, pattern in high_risk_patterns.items():
        if re.search(pattern, contract_text, re.IGNORECASE):
            risks_found.append(f"🔴 {risk_name.upper()}")
    
    # Check termination provisions
    sections = extract_section_blocks(contract_text)
    if 'term_termination' in sections:
        term_text = sections['term_termination']
        renewal_info = extract_renewal_terms(term_text)
        
        notice_days = renewal_info.get('notice_period_days')
        if notice_days is not None:
            days = int(notice_days)
            if days >= 90:
                risks_found.append(f"🟡 Long notice period: {days} days")
            elif days <= 30:
                risks_found.append(f"🟢 Reasonable notice: {days} days")
    
    if risks_found:
        result.append("\n🎯 Identified Risks:")
        result.extend([f"  • {risk}" for risk in risks_found])
    else:
        result.append("\n✅ No major risk flags identified in preliminary scan")
    
    result.append("\n💡 Recommendation: Full legal review recommended before execution")
    
    return "\n".join(result)


def generate_contract_summary(contract_text: str) -> str:
    """
    Generate an executive summary of a contract.
    """
    result = ["📊 EXECUTIVE SUMMARY\n" + "="*50]
    
    # Get metadata
    defined_terms = defined_terms_to_dict(extract_defined_terms(contract_text))
    
    # Contract basics
    result.append("\n📌 Key Information:")
    if 'Agreement' in defined_terms:
        result.append(f"  • Type: {defined_terms['Agreement']}")
    if 'Effective Date' in defined_terms:
        result.append(f"  • Effective: {defined_terms['Effective Date']}")
    if 'Initial Term' in defined_terms:
        result.append(f"  • Term: {defined_terms['Initial Term']}")
    
    # Parties
    party_terms = [k for k in defined_terms.keys() if k in ['Distributor', 'Supplier', 'Company', 'Customer', 'OEM']]
    if party_terms:
        result.append("\n🤝 Parties:")
        for party in party_terms:
            result.append(f"  • {party}: {defined_terms[party]}")
    
    # Quick risk check
    if 'exclusiv' in contract_text.lower():
        result.append("\n⚠️  Contains exclusivity provisions")
    
    if 'minimum' in contract_text.lower() and ('order' in contract_text.lower() or 'purchase' in contract_text.lower()):
        result.append("⚠️  Contains minimum purchase requirements")
    
    return "\n".join(result)


def map_contract_relationships(contract_info: str) -> str:
    """
    Map relationships between related contracts.
    
    Identifies:
    - MSA → SOW hierarchies
    - Amendments and their parent agreements
    - NDAs linked to business deals
    - Cross-references between documents
    """
    result = ["🔗 CONTRACT RELATIONSHIP MAPPING\n" + "="*50]
    
    # Extract contract references
    contract_refs = re.findall(r'(?:MSA|SOW|NDA|Agreement)\s*[#-]?\s*(\d{4}-\d{3,4}|\d+)', contract_info, re.IGNORECASE)
    
    if contract_refs:
        result.append("\n📎 Referenced Contracts:")
        for ref in set(contract_refs):
            result.append(f"  • {ref}")
    
    # Check for amendment language
    if re.search(r'amend(?:s|ment|ing)', contract_info, re.IGNORECASE):
        result.append("\n📝 Amendment Detected:")
        result.append("  • This document modifies an existing agreement")
        result.append("  • Review parent agreement for full context")
    
    # Check for SOW references to MSA
    if re.search(r'master\s+service\s+agreement', contract_info, re.IGNORECASE):
        result.append("\n🏢 MSA Relationship:")
        result.append("  • This SOW is governed by a Master Service Agreement")
        result.append("  • MSA terms apply unless specifically modified")
    
    return "\n".join(result)

# =============================================================================
# MANAGER AGENT
# =============================================================================

print("Creating CLM Manager Agent...")

clm_manager = Agent(
    name="clm_manager",
    model=Gemini(
        model="gemini-2.5-flash",  # Using the stable model that works with your API key
        retry_options=retry_config
    ),
    description="Contract Lifecycle Management system that analyzes agreements using specialized tools.",
    instruction="""You are a CLM system manager for a channel business handling various contract types:
    - MSA (Master Service Agreements)
    - OEM (Original Equipment Manufacturer) Agreements
    - Distributor Agreements
    - SOWs (Statements of Work)
    - NDAs (Non-Disclosure Agreements)
    - License Agreements
    
    Use your tools to analyze contracts:
    - extract_contract_metadata(): Extract parties, dates, terms, renewal provisions
    - analyze_contract_risks(): Identify legal, financial, operational risks
    - generate_contract_summary(): Create executive summaries
    - map_contract_relationships(): Find MSA-SOW links, amendments, dependencies
    
    IMPORTANT: Pay special attention to:
    1. Defined terms in parentheses like ("Effective Date"), ("Term"), ("Renewal Term")
    2. Renewal and termination provisions (notice periods, auto-renewal)
    3. Department ownership (IT, Sales, BizDev, Procurement, etc.)
    4. Hierarchical relationships (MSA → SOW → Amendments)
    
    Always call the appropriate tools and synthesize their outputs into a clear response.""",
    tools=[
        extract_contract_metadata,
        analyze_contract_risks,
        generate_contract_summary,
        map_contract_relationships
        # Note: google_search removed - Gemini 1.x doesn't support mixing it with custom tools
        # For contract analysis, we don't need web search anyway!
    ]
)

print("✅ CLM Manager created successfully!\n")

# =============================================================================
# DEMO SCENARIOS
# =============================================================================

DEMO_SCENARIOS = [
    {
        "name": "Distributor Agreement with Complex Renewal Terms",
        "query": """Analyze this distributor agreement and pay special attention to the renewal terms:
        
This Distributor Agreement ("Agreement") is entered as of March 1, 2014 (the "Effective Date") by and 
between Spectralink Corporation, a Delaware corporation ("Spectralink"), and Tel-E Connect Systems Ltd., 
an Ontario corporation ("Distributor").

11. TERM AND TERMINATION

11.1. Term. Unless otherwise terminated as set forth herein, this Agreement will begin on the Effective 
Date and will continue in effect until the end of one (1) year (the "Initial Term"). Thereafter, 
this Agreement will automatically renew for additional one (1) year terms (each, a "Renewal Term"), 
unless either party gives the other party at least sixty (60) days' notice of non-renewal prior to 
the end of the Initial Term or then-current Renewal Term.

11.2. Termination. Either party may terminate this Agreement if the other party breaches any 
provision of such breach and such breach is not cured within thirty (30) days after notice thereof.

Department: Sales with Procurement oversight
Minimum Order: $250K annually

Extract all metadata including the renewal notification requirements."""
    },
    {
        "name": "MSA-SOW Relationship Mapping",
        "query": """Map the relationships between these contracts:
        
1. Master Service Agreement MSA-2024-001 with TechVendor Inc., effective Jan 1, 2024
2. SOW-2024-015 under MSA-2024-001 for Cloud Infrastructure (Procurement/IT)
3. SOW-2024-022 under MSA-2024-001 for Security Services (IT/InfoSec)
4. Amendment #1 to SOW-2024-015 extending timeline by 3 months

Show the hierarchy and any dependencies."""
    },
    {
        "name": "OEM Agreement Risk Analysis",
        "query": """Assess risks in this OEM agreement:

OEM Manufacturing Agreement between TechCorp ("Customer") and GlobalMfg ("OEM")
Effective: January 1, 2025 (the "Effective Date")
Term: 3 years (the "Initial Term"), automatically renews for one (1) year terms (the "Renewal Term")
Minimum Purchase: $500K annually
Payment: Net 60 days
IP: Customer retains design IP; OEM owns manufacturing processes
Exclusivity: OEM cannot manufacture competing products for competitors during term and 18 months after
Liability: OEM liability capped at 12 months of fees paid
Notice: Either party may terminate with ninety (90) days' written notice

Department: Procurement with Product Management
        
Focus on financial and operational risks."""
    }
]

# =============================================================================
# MAIN EXECUTION
# =============================================================================

async def run_demo():
    """Run automated demo scenarios."""
    runner = InMemoryRunner(agent=clm_manager)
    
    for i, scenario in enumerate(DEMO_SCENARIOS, 1):
        print(f"\n{'='*70}")
        print(f"📋 Scenario {i}: {scenario['name']}")
        print(f"{'='*70}\n")
        
        await runner.run_debug(scenario['query'])
        
        if i < len(DEMO_SCENARIOS):
            print(f"\n⏳ Waiting 3 seconds before next scenario...")
            await asyncio.sleep(3)
    
    print(f"\n{'='*70}")
    print("✅ All demo scenarios completed!")
    print(f"{'='*70}")


async def interactive_mode():
    """Interactive CLI mode."""
    runner = InMemoryRunner(agent=clm_manager)
    session_id = None
    
    print(f"\n{'='*70}")
    print("💼 Contract Lifecycle Management - Interactive Mode")
    print(f"{'='*70}\n")
    
    print("Ask me to analyze contracts, extract metadata, assess risks, or map relationships.\n")
    print("Example queries:")
    print("  - 'Extract metadata from our NDA with Acme Corp'")
    print("  - 'What are the risks in this distributor agreement?'")
    print("  - 'Summarize our MSA with GlobalTech'")
    print("  - 'Map relationships between MSA-2024-005 and its SOWs'\n")
    print("Type 'demo' to run pre-built scenarios")
    print("Type 'quit' to exit")
    print(f"{'='*70}\n")
    
    while True:
        user_input = input("You: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("\n👋 CLM System shutting down. Goodbye!")
            break
        
        if user_input.lower() == 'demo':
            await run_demo()
            continue
        
        print("\n🤖 CLM Manager: Processing your request...\n")
        
        try:
            if session_id:
                await runner.run_debug(user_input, session_id=session_id)
            else:
                await runner.run_debug(user_input)
                session_id = "debug_session_id"
            
            print("\n" + "-"*70 + "\n")
        
        except Exception as e:
            print(f"❌ Error: {e}\n")


async def main():
    """Main entry point."""
    print("🚀 Starting CLM Multi-Agent System...\n")
    print("Choose mode:")
    print("1. Run demo scenarios (automated)")
    print("2. Interactive mode (ask questions)\n")
    
    choice = input("Enter 1 or 2: ").strip()
    
    if choice == "1":
        await run_demo()
    elif choice == "2":
        await interactive_mode()
    else:
        print("Invalid choice. Running demo scenarios...")
        await run_demo()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 CLM System interrupted. Goodbye!")
