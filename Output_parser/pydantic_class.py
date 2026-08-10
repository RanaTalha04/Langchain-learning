from pydantic import BaseModel, Field   

class Person(BaseModel):
    
    name: str = Field(description="Name of the person")
    age: int = Field(gt=18, description="Age of the person")
    city: str = Field(description="name of the city, the person belongs to.")