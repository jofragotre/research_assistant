from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.llms.ollama import Ollama


# TODO: Add more embedding and LLM model support for ollama

def get_embedding_function():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings


def get_llm():
    model = Ollama(model="llama3")
    return model
