from typing import Optional

from pydantic import BaseModel, Field


class DateOutput(BaseModel):
    """Date output"""
    date: str = Field(description="Date in YYYY-MM-DD format")

