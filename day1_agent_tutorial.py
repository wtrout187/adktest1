"""
Day 1: From Prompt to Action - ADK Tutorial
Adapted from Kaggle 5 Days of AI course for local development
"""

import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import google_search
from google.genai import types

# Load environment variables from .env file
env_path = Path(__file__).parent / "my_agent" / ".env"
load_dotenv(env_path)

# Verify API key is loaded
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY not found in .env file!")

print("✅ ADK components imported successfully.")
print(f"✅ API key loaded from: {env_path}")

# Configure Retry Options
# When working with LLMs, you may encounter transient errors like rate limits
# or temporary service unavailability. Retry options automatically handle these
# failures by retrying the request with exponential backoff.

retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,  # Initial delay before first retry (in seconds)
    http_status_codes=[429, 500, 503, 504]  # Retry on these HTTP errors
)

print("✅ Retry configuration set.")

# Define the Agent
# An agent can think, take actions, and observe the results of those actions
# to give you a better answer: Prompt -> Agent -> Thought -> Action -> Observation -> Final Answer

root_agent = Agent(
    name="helpful_assistant",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    description="A simple agent that can answer general questions.",
    instruction="You are a helpful assistant. Use Google Search for current info or if unsure.",
    tools=[google_search],
)

print("✅ Root Agent defined.")


# Create the Runner
# The Runner is the orchestrator that manages the conversation, sends messages
# to the agent, and handles its responses.

async def main():
    """Main function to run the agent examples."""
    
    runner = InMemoryRunner(agent=root_agent)
    print("✅ Runner created.")
    
    # Example 1: Ask about ADK
    print("\n" + "="*70)
    print("Example 1: Asking about Agent Development Kit")
    print("="*70)
    
    await runner.run_debug(
        "What is Agent Development Kit from Google? What languages is the SDK available in?"
    )
    
    # Example 2: Ask about current weather
    print("\n" + "="*70)
    print("Example 2: Asking about current weather")
    print("="*70)
    
    await runner.run_debug("What's the weather in London?")
    
    # Example 3: Your custom question
    print("\n" + "="*70)
    print("Example 3: Custom question - Latest tech news")
    print("="*70)
    
    await runner.run_debug("What are the top tech news stories today?")
    
    print("\n✅ All examples completed!")
    print("\nTip: The agent used Google Search to find current information.")
    print("Check the terminal output above to see the agent's reasoning process.")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
