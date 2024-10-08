#external 
from pydantic import BaseModel

class AddInput(BaseModel):
    int_1: int
    int_2: int

class CombineInput(BaseModel):
    string_1: str
    string_2: str

class GreetInput(BaseModel):
    name: str

class AddOutput(BaseModel):
    result: int

class CombineOutput(BaseModel):
    result: str

class GreetOutput(BaseModel):
    result: str

class GetEmbeddingParams(BaseModel):
    text: str

class EmbeddingOutput(BaseModel):
    embedding: list[float]

class UpsertInput(BaseModel):
    data: EmbeddingOutput
    metadata: dict[str, str]

class Query(BaseModel):
    text: str

