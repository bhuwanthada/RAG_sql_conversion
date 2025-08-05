from pydantic import BaseModel
from typing import List, Dict, Any


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    result: List[Dict[str, Any]]
    sql_query: str


class UserQuery(BaseModel):
    query: str