




# Initialize ChromaDB as Vector Store
vector_store = Chroma(
    collection_name="football_knowledge_base",
    embedding_function=embeddings
)

def create_embeddings(topic):
    