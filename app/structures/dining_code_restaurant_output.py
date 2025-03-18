from typing import Optional, List

from pydantic import BaseModel, Field

class DiningCodeRestaurantOutputList(BaseModel):
    """Dining code restaurant output list"""
    
    class DingCodeRestaurantOutput(BaseModel):
        """Dining code restaurant output"""
        class Menu(BaseModel):
            """Menu"""
            name: str = Field(description="Menu name")
            price: str = Field(description="Price")
        restaurant_id: str = Field(description="restaurant_id")
        restaurant_name: str = Field(description="Restaurant_name")
        category: str = Field(description="Category")
        keywords: Optional[List[str]] = Field(description="Keywords")
        menu: Optional[List[Menu]] = Field(default=None, description="Menu")
        score: int = Field(description="Score")
        review_cnt: int = Field(description="Review count")
        favorites_cnt: int = Field(description="Favorites count")
        recommend_cnt: int = Field(description="Recommend count")
        review: Optional[List[str]] = Field(description="Review")
    restaurants: List[DingCodeRestaurantOutput] = Field(description="Dining code restaurants")