from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional

class RouterType(str, Enum):
    """
    The 10 router types for query classification.
    TODO: Waiting for user to provide the actual 10 types.
    """
    TYPE_1 = "TYPE_1"
    TYPE_2 = "TYPE_2"
    TYPE_3 = "TYPE_3"
    TYPE_4 = "TYPE_4"
    TYPE_5 = "TYPE_5"
    TYPE_6 = "TYPE_6"
    TYPE_7 = "TYPE_7"
    TYPE_8 = "TYPE_8"
    TYPE_9 = "TYPE_9"
    TYPE_10 = "TYPE_10"

class RoutingDecision(BaseModel):
    """
    The validated JSON schema output by the LLM query router.
    """
    query: str = Field(..., description="The original natural language query")
    router_type: RouterType = Field(..., description="The classified intent of the query")
    requires_retrieval: bool = Field(..., description="Whether this query needs PostgreSQL retrieval")
    
    # Optional parameters depending on the router type
    keywords: Optional[list[str]] = Field(default=None, description="Key entities or words to search for")
    temporal_filter: Optional[str] = Field(default=None, description="Time or date constraints extracted from the query")
    spatial_filter: Optional[str] = Field(default=None, description="Geographic or location constraints extracted")
