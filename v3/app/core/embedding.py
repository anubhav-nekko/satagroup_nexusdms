"""
Embedding functionality for text vectorization and similarity search.
"""
import numpy as np
import faiss
import pickle
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from datetime import datetime

from app.config import (
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    INDEX_FILE,
    METADATA_FILE,
    S3_BUCKET
)
from app.core.aws_client import aws_client

class EmbeddingManager:
    """Manager for text embeddings and vector search."""
    
    def __init__(self):
        """Initialize embedding model and index."""
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.index = None
        self.metadata = []
        self.load_stores()
    
    def load_stores(self) -> None:
        """Load FAISS index and metadata from storage."""
        try:
            # Download from S3
            aws_client.download_file(INDEX_FILE, INDEX_FILE)
            aws_client.download_file(METADATA_FILE, METADATA_FILE)
            
            # Load into memory
            self.index = faiss.read_index(INDEX_FILE)
            with open(METADATA_FILE, "rb") as f:
                self.metadata = pickle.load(f)
        except Exception as e:
            print(f"Error loading stores: {e}")
            # Initialize new index if loading fails
            self.index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
            self.metadata = []
    
    def persist_stores(self) -> None:
        """Save FAISS index and metadata to storage."""
        try:
            # Save to local files
            faiss.write_index(self.index, INDEX_FILE)
            with open(METADATA_FILE, "wb") as f:
                pickle.dump(self.metadata, f)
            
            # Upload to S3
            aws_client.upload_file(INDEX_FILE, INDEX_FILE)
            aws_client.upload_file(METADATA_FILE, METADATA_FILE)
        except Exception as e:
            print(f"Error persisting stores: {e}")
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for text.
        
        Args:
            text: Input text
            
        Returns:
            Text embedding vector
        """
        return self.model.encode(text, normalize_embeddings=True)
    
    def add_to_index(self, text: str, metadata_info: Dict[str, Any]) -> int:
        """
        Add text embedding to index with metadata.
        
        Args:
            text: Text to embed
            metadata_info: Associated metadata
            
        Returns:
            Index of added embedding
        """
        # Generate embedding
        embedding = self.embed_text(text).reshape(1, -1)
        
        # Add to index
        self.index.add(embedding)
        
        # Add metadata
        metadata_entry = {
            **metadata_info,
            "text": text,
            "uploaded": datetime.utcnow().isoformat()
        }
        self.metadata.append(metadata_entry)
        
        # Persist changes
        self.persist_stores()
        
        return len(self.metadata) - 1
    
    def search(self, query: str, top_k: int = 20, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Search for similar texts.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filters: Metadata filters
            
        Returns:
            List of search results with metadata
        """
        if self.index.ntotal == 0:
            return []
        
        # Generate query embedding
        query_embedding = self.embed_text(query).reshape(1, -1)
        
        # Search index
        distances, indices = self.index.search(query_embedding, min(top_k * 5, self.index.ntotal))
        
        # Filter and format results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= len(self.metadata):
                continue
                
            meta = self.metadata[idx]
            
            # Apply filters if provided
            if filters:
                skip = False
                for key, value in filters.items():
                    if key not in meta or meta[key] != value:
                        skip = True
                        break
                if skip:
                    continue
            
            # Add result
            results.append({
                "score": float(dist),
                **meta
            })
            
            # Stop when we have enough results
            if len(results) >= top_k:
                break
        
        return results

# Create a singleton instance
embedding_manager = EmbeddingManager()
