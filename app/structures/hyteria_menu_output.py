from typing import List, Optional

from pydantic import BaseModel, Field

class HyteriaMenuOutput(BaseModel):
    """Hyteria Menu Output"""
    course_name: str = Field(description="Course name")
    menu_name: str = Field(description="Menu name")
    side_menus: Optional[List[str]] = Field(description="Side menus")
    menu_guide: str = Field(description="Menu guide")
    kcal: str = Field(description="Kcal")
    menu_origin: str = Field(description="Menu origin")
    avg_star: str = Field(description="Average star")
    sati_cnt: str = Field(description="Satisfaction count")

class HyteriaMenuOutputList(BaseModel):
    """Hyteria Menu Output List"""
    menus: List[HyteriaMenuOutput] = Field(description="Hyteria menus")