# imports
import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# load in the .env variables
load_dotenv()

# Get Embeddings Model
embeddings = OpenAIEmbeddings(model=os.getenv("TEXT_EMBEDDING_MODEL"))

# Initialize ChromaDB as Vector Store
vector_store = Chroma(
    collection_name="football_knowledge_base",
    embedding_function=embeddings
)