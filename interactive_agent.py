"""
Interactive Agent Demo
Ask questions and get real-time answers using Google Search
"""

import asyncio
from pathlib import Path
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import google_search
from google.genai import types

# Load environment variables
env_path = Path(__file__).parent / "my_agent" / ".env"
load_dotenv(env_path)

# Configure the agent
retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504]
)

agent = Agent(
    name="search_assistant",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    description="An assistant that can search the web for current information.",
    instruction="You are a helpful assistant. Always use Google Search to get accurate, up-to-date information.",
    tools=[google_search],
)

runner = InMemoryRunner(agent=agent)

print("="*70)
print("🤖 Interactive Agent with Google Search")
print("="*70)
print("\nThis agent can search the web to answer your questions!")
print("Type 'quit' or 'exit' to stop.\n")
print("Try asking:")
print("  - What's the weather in [city]?")
print("  - What's happening in the news today?")
print("  - Who won the [sport] championship?")
print("  - What's the stock price of [company]?")
print("="*70 + "\n")


async def chat():
    """Run an interactive chat session."""
    session_id = None
    
    while True:
        # Get user input
        user_input = input("You: ").strip()
        
        if not user_input:
            continue
            
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("\n👋 Goodbye!")
            break
        
        print(f"\n🔍 Agent: Searching and thinking...\n")
        
        try:
            # Run the agent with the user's question
            if session_id:
                # Continue existing session
                await runner.run_debug(user_input, session_id=session_id)
            else:
                # Create new session
                await runner.run_debug(user_input)
                # The session_id is created automatically, we can extract it if needed
                session_id = "debug_session_id"
            
            print("\n" + "-"*70 + "\n")
        
        except Exception as e:
            print(f"❌ Error: {e}\n")


if __name__ == "__main__":
    try:
        asyncio.run(chat())
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
