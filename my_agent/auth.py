"""Authentication and API layer.

This module provides a complete OAuth 2.0 authentication framework with support for:
1. AgentSpace Token authentication (for production deployments)
2. Personal Access Token / API Key authentication (for local development)
3. OAuth 2.0 with automatic token refresh (for multi-tenant applications)

Based on the ADK authentication patterns established in production agents.
"""

import functools
import logging
import os
from abc import ABC
from typing import Any, override

import httpx
from fastapi.openapi.models import OAuth2, OAuthFlowAuthorizationCode, OAuthFlows
from google.adk.auth.auth_credential import (
    AuthCredential,
    AuthCredentialTypes,
    OAuth2Auth,
)
from google.adk.auth.auth_tool import AuthConfig
from google.adk.auth.refresher.oauth2_credential_refresher import (
    OAuth2CredentialRefresher,
)
from google.adk.tools.tool_context import ToolContext
from httpx import BasicAuth

log = logging.getLogger(__name__)

# Token cache keys - customize these for your agent
TOKEN_CACHE = "agent_oauth_token"  # Choose a unique key for your agent
RESOURCE_CACHE = "agent_resource_cache"  # For caching API resources/instances


# ============================================================================
# Result Classes
# ============================================================================


class ResultsDict(dict[str, Any], ABC):
    """Base class for dictionary returned by tool functions."""

    def __init__(self, **kwargs: Any) -> None:
        """Construct a ResultsDict."""
        super().__init__(**kwargs)

    def is_success(self) -> bool:
        """Check if the results represent a success."""
        return False


class ResultsPending(ResultsDict):
    """Results when authentication is pending (OAuth flow initiated)."""

    def __init__(self) -> None:
        """Construct a pending results dictionary."""
        super().__init__(pending=True, message="Awaiting user authentication.")


class ResultsError(ResultsDict):
    """Results when there is an error (see message for details)."""

    def __init__(self, message: str) -> None:
        """Construct an error results dictionary."""
        super().__init__(status="error", message=message)


class ResultsSuccess(ResultsDict):
    """Results when operation succeeds (see data for payload)."""

    def __init__(self, data: dict[str, Any] | None = None) -> None:
        """Construct a success results dictionary."""
        super().__init__(status="success", data=data)

    @override
    def is_success(self) -> bool:
        """Check if results represent success."""
        return True


# ============================================================================
# OAuth Configuration
# ============================================================================


@functools.cache
def _get_auth_scheme_and_credential() -> tuple[OAuth2, AuthCredential]:
    """Get OAuth2 authentication scheme and credential from environment variables.

    This function is cached to avoid recreating the auth objects on every call.

    Returns:
        tuple: (OAuth2 scheme, AuthCredential) for ADK authentication

    Raises:
        ValueError: If required OAuth environment variables are not set
    """
    client_id = os.getenv("ADK_OAUTH_CLIENT_ID")
    client_secret = os.getenv("ADK_OAUTH_CLIENT_SECRET")
    if not client_secret or not client_id:
        msg = "ADK_OAUTH_CLIENT_ID and ADK_OAUTH_CLIENT_SECRET environment variables must be set"
        raise ValueError(msg)

    auth_uri = os.getenv("ADK_OAUTH_AUTH_URI")
    token_uri = os.getenv("ADK_OAUTH_TOKEN_URI")
    scopes = os.getenv("ADK_OAUTH_SCOPES")
    if not auth_uri or not token_uri or not scopes:
        msg = "ADK_OAUTH_AUTH_URI, ADK_OAUTH_TOKEN_URI and ADK_OAUTH_SCOPES environment variables must be set"
        raise ValueError(msg)

    auth_scheme = OAuth2(
        flows=OAuthFlows(
            authorizationCode=OAuthFlowAuthorizationCode(
                authorizationUrl=auth_uri,
                tokenUrl=token_uri,
                refreshUrl=token_uri,
                scopes={scope: f"Access to {scope}" for scope in scopes.split(" ")},
            ),
        ),
    )
    auth_credential = AuthCredential(
        auth_type=AuthCredentialTypes.OAUTH2,
        oauth2=OAuth2Auth(
            client_id=client_id,
            client_secret=client_secret,
            audience=os.getenv("ADK_OAUTH_AUDIENCE", None),
        ),
    )

    log.info("Auth schema: %s", auth_scheme)
    log.info("Auth credential: %s", auth_credential)

    return auth_scheme, auth_credential


# ============================================================================
# OAuth Token Management
# ============================================================================


async def _refresh_credentials(tool_context: ToolContext) -> ResultsDict:
    """Refresh OAuth credentials if needed.

    This function:
    1. Checks if cached credentials exist in tool_context.state
    2. Refreshes them if they're expired
    3. Initiates OAuth flow if no credentials exist

    Args:
        tool_context: The tool context containing session state

    Returns:
        ResultsDict: Success if credentials are ready, Pending if OAuth flow initiated
    """
    auth_scheme, auth_credential = _get_auth_scheme_and_credential()
    refresher = OAuth2CredentialRefresher()

    # Extract cached credentials from state
    auth_cred: str | None = tool_context.state.get(TOKEN_CACHE)
    creds: AuthCredential | None = (
        AuthCredential.model_validate_json(auth_cred) if auth_cred else None
    )

    if creds:
        try:
            if await refresher.is_refresh_needed(creds, auth_scheme):
                log.info("Refreshing OAuth credentials")
                tool_context.state[TOKEN_CACHE] = (
                    await refresher.refresh(creds, auth_scheme)
                ).model_dump_json()
            else:
                log.info("Credentials still valid")

            return ResultsSuccess()

        except Exception as e:
            log.exception("Error refreshing credentials", exc_info=e)
            tool_context.state[TOKEN_CACHE] = None

    # Try to get credentials from OAuth response
    creds = tool_context.get_auth_response(
        AuthConfig(
            auth_scheme=auth_scheme,
            raw_auth_credential=auth_credential,
        )
    )

    if creds:
        log.info("Got OAuth credentials, caching in state")
        tool_context.state[TOKEN_CACHE] = creds.model_dump_json()
        return ResultsSuccess()

    # No credentials yet - request OAuth flow
    log.info("Requesting OAuth authentication from user")
    tool_context.request_credential(
        AuthConfig(
            auth_scheme=auth_scheme,
            raw_auth_credential=auth_credential,
        ),
    )
    return ResultsPending()


# ============================================================================
# API Call Functions
# ============================================================================


async def api_call(  # noqa: PLR0913
    tool_context: ToolContext | None,
    endpoint: str,
    method: str,
    path: str,
    json_data: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
) -> ResultsDict:
    """Make an authenticated API call with automatic credential management.

    This function supports three authentication modes (in priority order):
    1. AgentSpace Token: Retrieved from AGENTSPACE_AUTH_ID in tool_context.state
    2. Personal Access Token: From API_USERNAME and API_TOKEN env vars
    3. OAuth 2.0: Full OAuth flow with automatic token refresh

    Args:
        tool_context: The tool context from the agent (required for OAuth/AgentSpace)
        endpoint: Base URL of the API (e.g., "https://api.example.com")
        method: HTTP method (GET, POST, PUT, DELETE, etc.)
        path: API path (e.g., "/v1/users")
        json_data: JSON payload for the request body
        params: URL query parameters

    Returns:
        ResultsDict: Success with response data, Error with message, or Pending for OAuth

    Example:
        >>> result = await api_call(
        ...     tool_context,
        ...     "https://api.example.com",
        ...     "GET",
        ...     "/v1/users",
        ...     params={"limit": 10}
        ... )
        >>> if result.is_success():
        ...     users = result["data"]
    """
    auth = None
    headers = {"Accept": "application/json", "Content-Type": "application/json"}

    # ========== Authentication Mode 1: AgentSpace Token ==========
    auth_id = os.getenv("AGENTSPACE_AUTH_ID")
    token: str | None = None
    if auth_id and tool_context and tool_context.state:
        log.info("Attempting AgentSpace token authentication")
        token = tool_context.state.get(auth_id)

    if token:
        log.info("Using AgentSpace token authentication")
        headers["Authorization"] = f"Bearer {token}"

    # ========== Authentication Mode 2: Personal Access Token (PAT) ==========
    elif os.getenv("API_USERNAME") and os.getenv("API_TOKEN"):
        log.info("Using Personal Access Token authentication")
        auth = BasicAuth(
            os.getenv("API_USERNAME", ""), os.getenv("API_TOKEN", "")
        )

    # ========== Authentication Mode 3: OAuth 2.0 ==========
    else:
        log.info("Using OAuth 2.0 authentication")

        if not tool_context or not tool_context.state:
            return ResultsError("tool_context required when using OAuth mode")

        # Refresh credentials if needed
        r = await _refresh_credentials(tool_context)
        if not is_success(r):
            return r

        # Extract access token from cached credentials
        creds = AuthCredential.model_validate_json(tool_context.state.get(TOKEN_CACHE))
        if not creds or not creds.oauth2 or not creds.oauth2.access_token:
            return ResultsError(
                f"No OAuth credentials available. Credentials state: {creds}"
            )

        headers["Authorization"] = f"Bearer {creds.oauth2.access_token}"

    # ========== Make API Request ==========
    try:
        async with httpx.AsyncClient(auth=auth, timeout=30) as client:
            log.info("Making %s request to %s%s", method, endpoint, path)
            response = await client.request(
                method,
                endpoint + path,
                headers=headers,
                params=params,
                json=json_data,
            )
            response.raise_for_status()
            return ResultsSuccess(response.json() if response.text else {})

    except httpx.HTTPStatusError as e:
        log.exception(
            "HTTP error calling API endpoint %s%s", endpoint, path, exc_info=e
        )
        return ResultsError(
            f"HTTP {e.response.status_code} error calling {endpoint}{path}: {e!s}"
        )
    except httpx.RequestError as e:
        log.exception("Request error calling API", exc_info=e)
        return ResultsError(f"Request error calling {endpoint}{path}: {e!s}")


# ============================================================================
# Helper Functions
# ============================================================================


def is_success(d: dict[str, Any]) -> bool:
    """Check whether a ResultsDict represents success.

    Args:
        d: Dictionary to check (typically a ResultsDict)

    Returns:
        bool: True if status is "success", False otherwise
    """
    return "status" in d and d["status"] == "success"


# ============================================================================
# Example: Multi-Resource Authentication (like Jira instances)
# ============================================================================


async def list_available_resources(tool_context: ToolContext | None) -> ResultsDict:
    """List available resources/instances for the authenticated user.

    This is an example function showing how to list multiple resources
    (like Jira instances, Salesforce orgs, etc.) that the user has access to.

    Customize this for your specific API:
    - Change the endpoint URL
    - Update the response parsing
    - Modify the cache key

    Args:
        tool_context: The tool context from the agent

    Returns:
        ResultsDict: Success with list of resources, Error, or Pending

    Example response:
        {
            "status": "success",
            "data": {
                "https://instance1.example.com": {
                    "id": "123",
                    "name": "Production Instance"
                },
                "https://instance2.example.com": {
                    "id": "456",
                    "name": "Staging Instance"
                }
            }
        }
    """
    # If single instance configured via env var, return it
    instance_url = os.getenv("API_BASE_URL")
    if instance_url:
        return ResultsSuccess(
            {
                instance_url: {
                    "id": None,
                    "name": "Default Instance",
                },
            },
        )

    # Otherwise, fetch from API (requires OAuth)
    if not tool_context or not tool_context.state:
        return ResultsError("tool_context required for multi-resource discovery")

    # Check cache first
    if not tool_context.state.get(RESOURCE_CACHE):
        # TODO: Customize this endpoint for your API
        # Example: Atlassian accessible-resources, Salesforce instances, etc.
        r = await api_call(
            tool_context,
            "https://api.example.com",  # Change this
            "GET",
            "/api/resources",  # Change this
        )
        if not is_success(r):
            return r

        # TODO: Customize this parsing for your API response format
        # Cache the resources
        tool_context.state[RESOURCE_CACHE] = {
            k["url"]: {
                "id": k["id"],
                "name": k["name"],
            }
            for k in r["data"]
        }

    # Return cached resources
    return ResultsSuccess(tool_context.state[RESOURCE_CACHE])
