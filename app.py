#builtin
import os
from dotenv import load_dotenv
import uuid

#external
from fastapi import FastAPI
from openai import AsyncOpenAI
from pinecone.grpc import PineconeGRPC as PineconeClient
from pinecone.grpc import GRPCIndex as Index

#internal
from ios import AddInput, CombineInput, GreetInput, AddOutput, CombineOutput, GreetOutput, GetEmbeddingParams, EmbeddingOutput, UpsertInput, Query, SearchOutput, Match

load_dotenv()

openai_client: AsyncOpenAI = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

pinecone_client: PineconeClient = PineconeClient(api_key=os.getenv("PINECONE_API_KEY"))
index: Index = pinecone_client.Index("oct7boil")

app: FastAPI = FastAPI()

@app.post("/add")
async def add(input: AddInput) -> AddOutput:
    return AddOutput(result=input.int_1 + input.int_2)

@app.post("/combine")
async def combine(combine_input: CombineInput) -> CombineOutput:
    return CombineOutput(result=combine_input.string_1 + combine_input.string_2)

@app.post("/greet")
async def greet(greet_input: GreetInput) -> GreetOutput:
    return GreetOutput(result=f"Hello {greet_input.name}")

async def get_embedding(params: GetEmbeddingParams) -> EmbeddingOutput:
    embedding: list[float] = await openai_client.embeddings.create(input=params.text, model="text-embedding-ada-002")
    return EmbeddingOutput(embedding=embedding.data[0].embedding)

def upsert_vectors(params: UpsertInput) -> None:
    vectors: list[list[float]] = []

    vectors.append({
        "id": str(uuid.uuid4()),
        "values": params.data.embedding,
        "metadata": params.metadata
    })

    index.upsert(vectors=vectors)

@app.post("/embed")
async def embed(text: GetEmbeddingParams) -> EmbeddingOutput:
    embeddingOut: EmbeddingOutput = await get_embedding(text)
    upsert_vectors(UpsertInput(data=embeddingOut, metadata={"text": text.text}))
    return embeddingOut

@app.post("/search")
async def search(query: Query) -> SearchOutput:
    embedding: list[float] = await openai_client.embeddings.create(
    input=query.text, 
    model="text-embedding-ada-002")

    embedding_output: EmbeddingOutput = EmbeddingOutput(embedding=embedding.data[0].embedding)

    results = index.query(
        vector=embedding_output.embedding,
        top_k=3,
        include_metadata=True
    )

    search_output: SearchOutput = SearchOutput(matches=[])

    for i in range(len(results['matches'])):
        match_data = results['matches'][i]
        match: Match = Match(
            id=match_data['id'], 
            metadata=match_data['metadata'] 
        )
        search_output.matches.append(match)
    
    return search_output