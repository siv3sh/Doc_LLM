"""
RAG Pipeline for multilingual document processing.
Handles document embedding, storage in Qdrant, and retrieval.
"""

import os
import uuid
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client.http import models
import numpy as np
from utils import clean_text, chunk_text

class RAGPipeline:
    """
    RAG Pipeline for multilingual document processing using Qdrant vector store.
    """
    
    def __init__(self, qdrant_url: str = None, collection_name: str = "multilingual_docs"):
        """
        Initialize RAG Pipeline.
        
        Args:
            qdrant_url: Qdrant server URL (default: localhost:6333)
            collection_name: Name of the Qdrant collection
        """
        self.qdrant_url = qdrant_url or os.getenv('QDRANT_URL', 'http://localhost:6333')
        self.collection_name = collection_name
        
        # Initialize multilingual embedding model
        self.embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        self.embedding_dimension = 384  # Dimension for the multilingual model
        
        # Initialize Qdrant client
        self.qdrant_client = None
        self.use_memory = False
        
        try:
            self.qdrant_client = QdrantClient(url=self.qdrant_url)
            # Test connection
            self.qdrant_client.get_collections()
            # Create collection if it doesn't exist
            self._ensure_collection_exists()
            print("Connected to Qdrant successfully")
        except Exception as e:
            print(f"Qdrant connection failed: {e}")
            print("Using in-memory storage")
            self.use_memory = True
            self.memory_storage = []  # In-memory storage for testing
            self.qdrant_client = None
    
    def _ensure_collection_exists(self):
        """Create Qdrant collection if it doesn't exist."""
        try:
            # Check if collection exists
            collections = self.qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                # Create collection
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dimension,
                        distance=Distance.COSINE
                    )
                )
                print(f"Created Qdrant collection: {self.collection_name}")
            else:
                print(f"Using existing Qdrant collection: {self.collection_name}")
                
        except Exception as e:
            print(f"Error ensuring collection exists: {e}")
            # Fallback: try to create collection
            try:
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dimension,
                        distance=Distance.COSINE
                    )
                )
            except Exception as create_error:
                print(f"Failed to create collection: {create_error}")
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            List[List[float]]: List of embedding vectors
        """
        try:
            embeddings = self.embedding_model.encode(texts, convert_to_tensor=False)
            return embeddings.tolist()
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            return []
    
    def process_document(self, text: str, filename: str, language: str) -> bool:
        """
        Process and store a document in the vector store.
        
        Args:
            text: Document text content
            filename: Name of the document file
            language: Detected language code
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Clean the text
            cleaned_text = clean_text(text) if text else text
            
            # Process even if text is minimal or empty
            if not cleaned_text or len(cleaned_text.strip()) == 0:
                print("Warning: Processing document with minimal/no text")
                cleaned_text = text if text else "[Empty Document]"
            
            # Chunk the text - accept even single character
            chunks = chunk_text(cleaned_text, chunk_size=500, overlap=50)
            
            # If no chunks generated, create at least one chunk with whatever we have
            if not chunks:
                print("Warning: No chunks generated, creating single chunk")
                chunks = [cleaned_text] if cleaned_text else ["[No content]"]
            
            # Generate embeddings for all chunks
            embeddings = self.generate_embeddings(chunks)
            
            if not embeddings:
                print("Failed to generate embeddings")
                return False
            
            # Prepare points for Qdrant
            points = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                point_id = str(uuid.uuid4())
                
                point = PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        'text': chunk,
                        'filename': filename,
                        'language': language,
                        'chunk_index': i,
                        'total_chunks': len(chunks)
                    }
                )
                points.append(point)
            
            # Store points in Qdrant or memory
            if self.use_memory or self.qdrant_client is None:
                # Store in memory
                for point in points:
                    self.memory_storage.append({
                        'id': point.id,
                        'vector': point.vector,
                        'payload': point.payload
                    })
                print(f"Successfully stored {len(points)} chunks in memory for {filename}")
            else:
                # Store in Qdrant
                self.qdrant_client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                print(f"Successfully stored {len(points)} chunks for {filename}")
            
            return True
            
        except Exception as e:
            print(f"Error processing document {filename}: {e}")
            return False
    
    def retrieve_context(self, query: str, top_k: int = 3, language: str = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context chunks for a query.
        
        Args:
            query: User query string
            top_k: Number of top results to retrieve
            language: Optional language filter
            
        Returns:
            List[Dict[str, Any]]: List of relevant chunks with metadata
        """
        try:
            # Generate embedding for the query
            query_embedding = self.generate_embeddings([query])
            
            if not query_embedding:
                return []
            
            if self.use_memory or self.qdrant_client is None:
                # Search in memory storage
                from sklearn.metrics.pairwise import cosine_similarity
                import numpy as np
                
                # Filter by language if specified
                filtered_storage = self.memory_storage
                if language:
                    filtered_storage = [item for item in self.memory_storage 
                                      if item['payload'].get('language') == language]
                
                if not filtered_storage:
                    return []
                
                # Calculate similarities
                query_vector = np.array(query_embedding[0]).reshape(1, -1)
                similarities = []
                
                for item in filtered_storage:
                    item_vector = np.array(item['vector']).reshape(1, -1)
                    similarity = cosine_similarity(query_vector, item_vector)[0][0]
                    similarities.append((similarity, item))
                
                # Sort by similarity and get top_k
                similarities.sort(key=lambda x: x[0], reverse=True)
                top_results = similarities[:top_k]
                
                # Format results
                context_chunks = []
                for score, item in top_results:
                    context_chunks.append({
                        'text': item['payload']['text'],
                        'filename': item['payload']['filename'],
                        'language': item['payload']['language'],
                        'chunk_index': item['payload']['chunk_index'],
                        'score': float(score)
                    })
                
                return context_chunks
            else:
                # Search in Qdrant
                search_params = {
                    'collection_name': self.collection_name,
                    'query_vector': query_embedding[0],
                    'limit': top_k,
                    'with_payload': True
                }
                
                # Add language filter if specified
                if language:
                    search_params['query_filter'] = models.Filter(
                        must=[
                            models.FieldCondition(
                                key="language",
                                match=models.MatchValue(value=language)
                            )
                        ]
                    )
                
                # Search in Qdrant
                search_results = self.qdrant_client.search(**search_params)
                
                # Format results
                context_chunks = []
                for result in search_results:
                    context_chunks.append({
                        'text': result.payload['text'],
                        'filename': result.payload['filename'],
                        'language': result.payload['language'],
                        'chunk_index': result.payload['chunk_index'],
                        'score': result.score
                    })
                
                return context_chunks
            
        except Exception as e:
            print(f"Error retrieving context: {e}")
            return []
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the current collection.
        
        Returns:
            Dict[str, Any]: Collection information
        """
        try:
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            return {
                'name': self.collection_name,
                'vectors_count': collection_info.vectors_count,
                'status': collection_info.status,
                'config': collection_info.config
            }
        except Exception as e:
            print(f"Error getting collection info: {e}")
            return {}
    
    def delete_document(self, filename: str) -> bool:
        """
        Delete all chunks belonging to a specific document.
        
        Args:
            filename: Name of the document to delete
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Search for all points with the given filename
            search_results = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="filename",
                            match=models.MatchValue(value=filename)
                        )
                    ]
                ),
                limit=10000,  # Large limit to get all chunks
                with_payload=True
            )
            
            if search_results[0]:  # Check if any points found
                point_ids = [point.id for point in search_results[0]]
                
                # Delete the points
                self.qdrant_client.delete(
                    collection_name=self.collection_name,
                    points_selector=models.PointIdsList(points=point_ids)
                )
                
                print(f"Deleted {len(point_ids)} chunks for {filename}")
                return True
            else:
                print(f"No chunks found for {filename}")
                return False
                
        except Exception as e:
            print(f"Error deleting document {filename}: {e}")
            return False
    
    def clear_collection(self) -> bool:
        """
        Clear all data from the collection.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.qdrant_client.delete_collection(self.collection_name)
            self._ensure_collection_exists()
            print(f"Cleared collection: {self.collection_name}")
            return True
        except Exception as e:
            print(f"Error clearing collection: {e}")
            return False
