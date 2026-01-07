"""Custom tools for the agent.

This module defines all the tools available to the agent. Tools are functions that
the agent can call to perform specific operations.
"""

import logging
from typing import Any

from google.adk.tools.tool_context import ToolContext

log = logging.getLogger(__name__)


async def example_tool(
    tool_context: ToolContext | None,
    message: str,
) -> dict[str, Any]:
    """An example tool that echoes a message.

    This is a simple example tool to demonstrate the pattern. Replace this
    with your own tools that provide real functionality for your agent.

    Args:
        tool_context: The tool context from the agent (contains session state, etc.).
        message: The message to echo back to the user.

    Returns:
        dict: A dictionary containing the status and echoed message.
              Example: {"status": "success", "message": "Echo: Hello!"}
    """
    log.info(f"Example tool called with message: {message}")

    return {
        "status": "success",
        "message": f"Echo: {message}",
    }


# ============================================================================
# Example: Authenticated API Tool
# ============================================================================
# This example shows how to use the auth.py framework to make authenticated
# API calls. Uncomment and customize for your API.
#
# from .auth import api_call, is_success
#
# async def get_user_info(
#     tool_context: ToolContext | None,
#     user_id: str,
# ) -> dict[str, Any]:
#     """Get user information from the API.
#
#     This tool demonstrates how to use the authentication framework to make
#     API calls with automatic credential management (AgentSpace, PAT, or OAuth).
#
#     Args:
#         tool_context: The tool context from the agent (required for auth).
#         user_id: The ID of the user to retrieve.
#
#     Returns:
#         dict: User information or error message.
#               Example: {"status": "success", "data": {"name": "John", "email": "john@example.com"}}
#     """
#     # Make authenticated API call
#     result = await api_call(
#         tool_context,
#         endpoint=os.getenv("API_BASE_URL", "https://api.example.com"),
#         method="GET",
#         path=f"/v1/users/{user_id}",
#     )
#
#     # Check if successful
#     if is_success(result):
#         log.info(f"Successfully retrieved user {user_id}")
#         return result
#     else:
#         log.error(f"Failed to retrieve user {user_id}: {result.get('message', 'Unknown error')}")
#         return result
#
#
# Add your custom tools below this line
# Example:
#
# async def my_custom_tool(
#     tool_context: ToolContext | None,
#     param1: str,
#     param2: int = 10,
# ) -> dict[str, Any]:
#     """Description of what your tool does.
#
#     Args:
#         tool_context: The tool context from the agent.
#         param1: Description of param1.
#         param2: Description of param2 (default: 10).
#
#     Returns:
#         dict: Description of the return value.
#     """
#     # Your tool implementation here
#     result = perform_operation(param1, param2)
#
#     return {
#         "status": "success",
#         "data": result,
#     }
