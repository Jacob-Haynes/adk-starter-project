"""Agent definition.

This module defines the root agent for the application, configured with
environment variables and callback functions for state management.
"""

import logging
import os

import google.cloud.logging
from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.genai import types

from .prompts import return_agent_instructions
from .tools import example_tool

# --- Logging Configuration ---
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Optional: Google Cloud Logger integration
if os.getenv("ENABLE_CLOUD_LOGGING", "false").lower() == "true":
    client = google.cloud.logging.Client()
    client.setup_logging()
    logger.info("Cloud logging enabled")


def setup_before_agent_call(callback_context: CallbackContext) -> None:
    """Callback function run before each agent invocation.

    Use this to initialize state, validate context, or perform setup tasks.

    Args:
        callback_context: The callback context containing session state and metadata.
    """
    # Add custom initialization logic here if needed
    # Example: callback_context.state["user_id"] = get_user_id()
    pass


# --- Agent Definition ---

root_agent = Agent(
    model=os.getenv("AGENT_MODEL", "gemini-2.5-flash"),
    name="my_agent",
    instruction=return_agent_instructions(),
    tools=[example_tool],
    before_agent_callback=setup_before_agent_call,
    generate_content_config=types.GenerateContentConfig(
        temperature=float(os.getenv("AGENT_TEMPERATURE", "0.3")),
    ),
)
