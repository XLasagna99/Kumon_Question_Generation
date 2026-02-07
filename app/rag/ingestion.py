from langchain_text_splitters import CharacterTextSplitter

# Read in State of the Union Address File
with open("2024_state_of_the_union.txt") as f:
    state_of_the_union = f.read()

# Initialize Text Splitter
text_splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)

# Create Documents (Chunks) From File
texts = text_splitter.create_documents([state_of_the_union])


def rag_document_ingestion(textfile):
    """Ingest documents for RAG pipeline."""
    with open(textfile) as f:
        document = f.read()
    
    # Initialize Text Splitter
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    # Create Documents (Chunks) From File
    texts = text_splitter.create_documents([document])
    return texts
    