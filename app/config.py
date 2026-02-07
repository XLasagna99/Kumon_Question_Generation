import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("MATH_PROJECT_API_KEY_OPENAI")
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
