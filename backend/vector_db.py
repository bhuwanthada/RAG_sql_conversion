import chromadb
from chromadb.utils import embedding_functions
from langchain_community.embeddings import SentenceTransformerEmbeddings
from sentence_transformers import SentenceTransformer
import os

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "chroma_db")  # Default path
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2") # Example embedding model


class VectorDB:
    def __init__(self):
        # self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        self.embedding_model = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)

    def get_or_create_collection(self, collection_name):
        """Gets or creates a ChromaDB collection."""
        return self.client.get_or_create_collection(name=collection_name, embedding_function=self.embedding_model)

    def add_data(self, collection, texts, metadatas, ids):
        """Adds data to a ChromaDB collection."""
        collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )

    def query(self, collection, query_texts, n_results=5):
        """Queries a ChromaDB collection."""
        results = collection.query(
            query_texts=query_texts,
            n_results=n_results,
        )
        return results