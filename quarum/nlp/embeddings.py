"""
Vector embeddings module for semantic search and similarity.

This module provides functionality for creating and querying vector
embeddings of text chunks to support semantic retrieval and analysis.
"""
import numpy as np
from numpy.linalg import norm

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document


class VectorEmbeddings:
    """
    Manages vector embeddings for semantic similarity and retrieval.

    This class handles creating vector embeddings from text chunks,
    storing them in a vector database, and retrieving semantically
    similar chunks for analysis.
    """

    def __init__(self, api_key: str, embedding_model: str = "text-embedding-ada-002"):
        """
        Initialize the vector embeddings manager.

        Args:
            api_key: OpenAI API key for embedding generation
            embedding_model: Name of the embedding model to use
        """
        self.api_key = api_key
        self.embedding_model = embedding_model
        self.embeddings = None
        self.vector_store = None
        self.retriever = None
        self._setup_embeddings()

    def _setup_embeddings(self) -> None:
        """Initialize the embedding model and prepare for use."""
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=self.api_key, model=self.embedding_model
        )

    def create_vector_store(self, documents: list[Document]) -> None:
        """
        Create a vector store from documents.

        Args:
            documents: list of Document objects to embed and store
        """
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 5}
        )

    def similarity_search(self, query: str, k: int = 5) -> list[Document]:
        """
        Perform a similarity search using the query.

        Args:
            query: The query text to search for
            k: Number of results to return

        Returns:
            list of Document objects similar to the query
        """
        if not self.retriever:
            raise ValueError(
                "Vector store not initialized. Call create_vector_store first."
            )

        try:
            return self.retriever.invoke(query)[:k]
        except Exception as e:
            print(f"Error retrieving documents: {str(e)}")
            return []

    def add_documents(self, documents: list[Document]) -> None:
        """
        Add documents to an existing vector store.

        Args:
            documents: list of Document objects to add
        """
        if not self.vector_store:
            self.create_vector_store(documents)
        else:
            self.vector_store.add_documents(documents)

    def get_document_similarity(self, doc1: Document, doc2: Document) -> float:
        """
        Calculate similarity between two documents.

        Args:
            doc1: First document
            doc2: Second document

        Returns:
            Similarity score (0.0-1.0)
        """
        # Get embeddings
        embedding1 = self.embeddings.embed_documents([doc1.page_content])[0]
        embedding2 = self.embeddings.embed_documents([doc2.page_content])[0]

        # Convert to numpy arrays for calculation
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        # Calculate cosine similarity
        similarity = np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))

        return float(similarity)
