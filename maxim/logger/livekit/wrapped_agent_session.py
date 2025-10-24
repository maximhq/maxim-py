"""
MaximWrappedAgentSession - A wrapper around LiveKit's AgentSession with additional parameters.

This module provides a MaximWrappedAgentSession class that extends the original AgentSession
with the ability to accept additional parameters for Maxim integration. This is useful for 
passing custom metadata (like session names and tags) that can be accessed during session 
instrumentation.

Usage:
    from maxim.logger.livekit import MaximWrappedAgentSession
    
    # Instead of:
    # session = AgentSession(turn_detection="manual")
    
    # Use:
    session = MaximWrappedAgentSession(
        turn_detection="manual",
        maxim_params={
            "session_name": "my-custom-session",
            "tags": {"user_id": "123", "department": "sales"}
        }
    )
"""

from typing import Any, Dict, Optional, TypedDict
from livekit.agents import AgentSession


class MaximParams(TypedDict, total=False):
    """Type definition for Maxim parameters.
    
    Attributes:
        session_name: Optional custom name for the session
        tags: Optional dictionary of key-value tags to attach to the session
    """
    session_name: str
    tags: Dict[str, Any]


class MaximWrappedAgentSession(AgentSession):
    """
    A wrapper around LiveKit's AgentSession that accepts additional parameters.
    
    This class extends AgentSession to allow passing custom parameters for Maxim
    integration that can be accessed during session instrumentation, particularly 
    in the intercept_session_start function.
    
    Args:
        *args: Positional arguments passed to the original AgentSession
        maxim_params: Optional MaximParams with session_name and tags
        **kwargs: Keyword arguments passed to the original AgentSession
    
    Example:
        session = MaximWrappedAgentSession(
            turn_detection="manual",
            maxim_params={
                "session_name": "customer-support-session",
                "tags": {"user_id": "user_123", "department": "sales"}
            }
        )
    """
    
    def __init__(self, *args, maxim_params: Optional[MaximParams] = None, **kwargs):
        """
        Initialize the MaximWrappedAgentSession.
        
        Args:
            *args: Positional arguments for AgentSession
            maxim_params: MaximParams with optional session_name and tags
            **kwargs: Keyword arguments for AgentSession
        """
        # Initialize the parent AgentSession with original parameters
        super().__init__(*args, **kwargs)
        
        # Store the additional parameters
        self._maxim_params: MaximParams = maxim_params or {}
    
    def get_maxim_param(self, key: str, default: Any = None) -> Any:
        """
        Get a specific Maxim parameter value.
        
        Args:
            key: The parameter key to retrieve
            default: Default value if key is not found
            
        Returns:
            The parameter value or default if not found
        """
        return self._maxim_params.get(key, default)
    
    def get_all_maxim_params(self) -> MaximParams:
        """
        Get all Maxim parameters.
        
        Returns:
            Copy of all custom parameters
        """
        return self._maxim_params.copy()
    