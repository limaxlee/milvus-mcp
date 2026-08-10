from pydantic import BaseModel


class ListDatabasesResult(BaseModel):
    databases: list[str] = []
