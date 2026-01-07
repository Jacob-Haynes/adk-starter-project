"""My Agent Package.

This module makes the root_agent available at package level.
Required for ADK CLI auto-discovery.
"""

from . import agent

__all__ = ["agent"]
