"""
RAG Engine Module
-----------------
Core retrieval and generation engine for the Knowledge-Base Search Engine.
Combines semantic search with metadata-based re-ranking and LLM synthesis.
"""

import os
import logging
import time
from typing import List, Dict, Any, Optional

# Core dependencies
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from groq import Groq

# Add parent directory to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    EMBEDDING_MODEL,
    VECTOR_DB_DIR,
    TOP_K_RESULTS,
    RELEVANCE_THRESHOLD,
    GROQ_API_KEY,
    GROQ_MODEL,
    MAX_TOKENS,
    TEMPERATURE,
    SYSTEM_PROMPT
)

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    logger.addHandler(handler)


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------
class RAGEngineError(Exception):
    """Base exception for RAG engine errors."""
    pass


class RetrievalError(RAGEngineError):
    """Raised when document retrieval fails."""
    pass


class GenerationError(RAGEngineError):
    """Raised when answer generation fails."""
    pass


# ---------------------------------------------------------------------------
# RAG Engine
# ---------------------------------------------------------------------------
class RAGEngine:
    """
    Intelligent RAG engine for programming documentation.
    Features semantic search, metadata-based re-ranking, and grounded generation.
    """
    
    def __init__(self, collection_name: str = "programming_docs"):
        """Initialize RAG engine with vector DB and LLM."""
        self.collection_name = collection_name
        self.embedding_model = None
        self.vector_db = None
        self.collection = None
        self.llm_client = None
        
        self._initialize_components()
        logger.info(f"✅ RAG Engine initialized: {collection_name}")

    # -----------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------
    def _initialize_components(self) -> None:
        """Initialize all components."""
        try:
            # 1. Embedding model
            logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
            logger.info("✅ Embedding model loaded")
            
            # 2. Vector database
            self._initialize_vector_db()
            
            # 3. LLM client
            if GROQ_API_KEY:
                self.llm_client = Groq(api_key=GROQ_API_KEY)
                logger.info(f"✅ LLM client initialized: {GROQ_MODEL}")
            else:
                logger.warning("⚠️  GROQ_API_KEY not found - generation will be disabled")
                
        except Exception as e:
            logger.exception(f"Failed to initialize RAG engine: {e}")
            raise RAGEngineError(f"Initialization failed: {e}") from e

    def _initialize_vector_db(self) -> None:
        """Initialize ChromaDB."""
        try:
            self.vector_db = chromadb.PersistentClient(
                path=VECTOR_DB_DIR,
                settings=Settings(anonymized_telemetry=False)
            )
            
            self.collection = self.vector_db.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Programming documentation chunks"}
            )
            
            logger.info(f"✅ Vector DB initialized: {VECTOR_DB_DIR}")
            logger.info(f"📊 Current chunks in DB: {self.collection.count()}")
            
        except Exception as e:
            logger.exception(f"Vector DB initialization failed: {e}")
            raise RAGEngineError(f"Vector DB setup failed: {e}") from e

    # -----------------------------------------------------------------------
    # Document Management
    # -----------------------------------------------------------------------
    def add_documents(self, chunks: List[Dict[str, Any]], replace: bool = False) -> int:
        """
        Add processed chunks to vector database.
        
        Args:
            chunks: List of chunks from document processor
            replace: If True, clear existing collection first
            
        Returns:
            Number of chunks added
        """
        if not chunks:
            logger.warning("No chunks provided")
            return 0
            
        try:
            # Optional: Replace existing collection
            if replace:
                logger.info("Replacing existing collection...")
                self.vector_db.delete_collection(self.collection_name)
                self.collection = self.vector_db.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Programming documentation chunks"}
                )
            
            # Prepare data
            documents = []
            metadatas = []
            ids = []
            
            for chunk in chunks:
                # Validate chunk structure
                if "chunk_id" not in chunk:
                    logger.warning(f"Chunk missing chunk_id, skipping...")
                    continue
                
                documents.append(chunk["content"])
                metadatas.append(chunk["metadata"])
                ids.append(chunk["chunk_id"])
            
            if not documents:
                logger.warning("No valid chunks to add")
                return 0
            
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(documents)} chunks...")
            start_time = time.time()
            embeddings = self.embedding_model.encode(
                documents,
                show_progress_bar=True,
                batch_size=32
            ).tolist()
            embed_time = time.time() - start_time
            logger.info(f"✅ Embeddings generated in {embed_time:.2f}s")
            
            # Add to collection
            self.collection.add(
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"✅ Added {len(documents)} chunks to vector DB")
            return len(documents)
            
        except Exception as e:
            logger.exception(f"Failed to add documents: {e}")
            raise RAGEngineError(f"Document addition failed: {e}") from e

    def clear_collection(self) -> None:
        """Clear all documents from collection."""
        try:
            self.vector_db.delete_collection(self.collection_name)
            self.collection = self.vector_db.create_collection(
                name=self.collection_name,
                metadata={"description": "Programming documentation chunks"}
            )
            logger.info("✅ Collection cleared")
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            raise RAGEngineError(f"Clear failed: {e}") from e

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        try:
            count = self.collection.count()
            return {
                "collection_name": self.collection_name,
                "total_chunks": count,
                "vector_db_path": VECTOR_DB_DIR,
                "embedding_model": EMBEDDING_MODEL,
                "llm_model": GROQ_MODEL
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}

    # -----------------------------------------------------------------------
    # Search & Retrieval
    # -----------------------------------------------------------------------
    def search(self, query: str, filters: Optional[Dict] = None, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Semantic search with optional metadata filtering.
        
        Args:
            query: Search query
            filters: Optional metadata filters
            top_k: Number of results (default: TOP_K_RESULTS)
            
        Returns:
            List of relevant chunks with scores
        """
        if top_k is None:
            top_k = TOP_K_RESULTS
            
        try:
            # Preprocess query
            enhanced_query = self._preprocess_query(query)
            
            # Generate embedding
            query_embedding = self.embedding_model.encode([enhanced_query]).tolist()[0]
            
            # Search parameters
            search_kwargs = {
                "query_embeddings": [query_embedding],
                "n_results": min(top_k * 2, 20),  # Retrieve more for re-ranking
            }
            
            if filters:
                search_kwargs["where"] = filters
            
            # Perform search
            results = self.collection.query(**search_kwargs)
            
            # Process results
            relevant_chunks = self._process_search_results(results, query)
            
            # Re-rank
            if relevant_chunks:
                relevant_chunks = self._rerank_chunks(query, relevant_chunks)
            
            # Limit to top_k
            relevant_chunks = relevant_chunks[:top_k]
            
            logger.info(f"Search '{query}': {len(relevant_chunks)} chunks retrieved")
            return relevant_chunks
            
        except Exception as e:
            logger.exception(f"Search failed for '{query}': {e}")
            raise RetrievalError(f"Search failed: {e}") from e

    def _preprocess_query(self, query: str) -> str:
        """Enhance query for better retrieval."""
        # Expand common abbreviations
        expansions = {
            " api ": " API application programming interface ",
            " db ": " database ",
            " auth ": " authentication ",
            " config ": " configuration ",
            " async ": " asynchronous ",
            " sync ": " synchronous ",
        }
        
        enhanced = " " + query.lower() + " "
        for abbrev, full in expansions.items():
            enhanced = enhanced.replace(abbrev, full)
        
        return enhanced.strip()

    def _process_search_results(self, results: Dict, query: str) -> List[Dict[str, Any]]:
        """Process raw search results."""
        relevant_chunks = []
        
        if not results.get("documents") or not results["documents"][0]:
            return relevant_chunks
        
        for i, (doc, metadata, doc_id, distance) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["ids"][0],
            results.get("distances", [[]])[0] if results.get("distances") else [0.0] * len(results["documents"][0])
        )):
            # Convert distance to similarity
            # ChromaDB uses L2 distance, convert to similarity score
            similarity_score = max(0.0, 1.0 - (distance / 2.0))
            
            # Apply relevance threshold
            if similarity_score >= RELEVANCE_THRESHOLD:
                relevant_chunks.append({
                    "id": doc_id,
                    "content": doc,
                    "metadata": metadata,
                    "similarity_score": round(similarity_score, 4),
                    "distance": round(distance, 4),
                    "rank": i + 1
                })
        
        return relevant_chunks

    def _rerank_chunks(self, query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Re-rank chunks using metadata signals.
        This is a key differentiator - using document processor metadata!
        """
        if not chunks:
            return chunks
        
        query_lower = query.lower()
        
        for chunk in chunks:
            base_score = chunk["similarity_score"]
            metadata = chunk["metadata"]
            
            # Boost for importance
            if metadata.get("importance") == "high":
                base_score *= 1.3
            
            # Boost for code if query asks for examples
            if any(keyword in query_lower for keyword in ["example", "code", "how to", "show me"]):
                if metadata.get("has_code", False):
                    base_score *= 1.25
            
            # Boost for API docs if query mentions API
            if any(keyword in query_lower for keyword in ["api", "endpoint", "request", "response"]):
                if metadata.get("content_type") == "api_documentation":
                    base_score *= 1.2
            
            # Boost for setup guides if query is about installation/setup
            if any(keyword in query_lower for keyword in ["install", "setup", "configure", "getting started"]):
                if metadata.get("content_type") == "setup_guide":
                    base_score *= 1.2
            
            chunk["reranked_score"] = min(1.0, base_score)
            chunk["original_rank"] = chunk["rank"]
        
        # Sort by reranked score
        reranked = sorted(chunks, key=lambda x: x["reranked_score"], reverse=True)
        
        # Update ranks
        for i, chunk in enumerate(reranked, 1):
            chunk["rank"] = i
        
        return reranked

    # -----------------------------------------------------------------------
    # Generation
    # -----------------------------------------------------------------------
    def generate_answer(self, query: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate answer using LLM with retrieved context.
        
        Args:
            query: User's question
            context_chunks: Retrieved relevant chunks
            
        Returns:
            Generated answer with metadata
        """
        if not self.llm_client:
            raise GenerationError("LLM client not initialized - check GROQ_API_KEY")
            
        if not context_chunks:
            return self._handle_no_context(query)
        
        try:
            # Prepare context
            context_text = self._prepare_context(context_chunks)
            
            # Create prompt
            user_message = f"""
Using the provided documentation context, answer the user's question. Follow these strict rules:
1. Identify the most relevant chunks to answer the question.
2. Formulate a direct and concise answer based ONLY on the information in those chunks.
3. Immediately after each piece of information or fact in your answer, add a citation in the format (Source [chunk_number]). The chunk_number corresponds to its position in the list below.
4. If a piece of information is found in multiple chunks, cite all of them, e.g., (Source [1, 3]).
5. If the documentation does not contain the answer, state "The provided documentation does not contain information on this topic." Do NOT hallucinate or infer an answer.
6. When relevant and available, include code examples and cite their source.

---
Question: {query}

Context from documentation:
{context_text}
---
Final Answer:
"""

            # Generate answer
            start_time = time.time()
            response = self.llm_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE
            )
            generation_time = time.time() - start_time
            
            answer = response.choices[0].message.content
            
            # Calculate confidence
            confidence = self._calculate_confidence(context_chunks)
            
            # Extract sources
            sources = self._extract_sources(context_chunks)
            
            result = {
                "answer": answer,
                "sources": sources,
                "confidence": confidence,
                "chunks_used": len(context_chunks),
                "model": GROQ_MODEL,
                "generation_time_ms": round(generation_time * 1000, 2)
            }
            
            logger.info(f"Answer generated in {generation_time:.2f}s")
            return result
            
        except Exception as e:
            logger.exception(f"Answer generation failed: {e}")
            raise GenerationError(f"Generation failed: {e}") from e

    def query(self, question: str, filters: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Complete RAG pipeline: search + generate.
        
        Args:
            question: User's question
            filters: Optional metadata filters
            
        Returns:
            Complete response with answer and metadata
        """
        logger.info(f"Processing query: {question}")
        start_time = time.time()
        
        try:
            # Search
            search_start = time.time()
            context_chunks = self.search(question, filters)
            search_time = time.time() - search_start
            
            # Generate
            if context_chunks:
                result = self.generate_answer(question, context_chunks)
            else:
                result = self._handle_no_context(question)
            
            # Add metadata
            total_time = time.time() - start_time
            result.update({
                "question": question,
                "chunks_retrieved": len(context_chunks),
                "filters_applied": bool(filters),
                "search_time_ms": round(search_time * 1000, 2),
                "total_time_ms": round(total_time * 1000, 2)
            })
            
            logger.info(f"Query completed in {total_time:.2f}s")
            return result
            
        except Exception as e:
            logger.exception(f"Query failed: {e}")
            raise RAGEngineError(f"Query failed: {e}") from e

    # -----------------------------------------------------------------------
    # Utility Methods
    # -----------------------------------------------------------------------
    def _prepare_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Prepare context string for LLM."""
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            source = chunk["metadata"].get("document_name", "Unknown")
            content_type = chunk["metadata"].get("content_type", "general")
            relevance = chunk.get("similarity_score", 0)
            
            context_parts.append(
                f"[Source {i}: {source} | Type: {content_type} | Relevance: {relevance:.2f}]\n"
                f"{chunk['content']}\n"
            )
        
        return "\n---\n".join(context_parts)

    def _calculate_confidence(self, chunks: List[Dict[str, Any]]) -> float:
        """Calculate confidence score based on retrieval quality."""
        if not chunks:
            return 0.0
        
        # Average similarity scores
        sim_scores = [c.get("similarity_score", 0) for c in chunks]
        avg_similarity = sum(sim_scores) / len(sim_scores)
        
        # Boost for high-importance chunks
        important_chunks = [c for c in chunks if c["metadata"].get("importance") == "high"]
        if important_chunks:
            boost = min(0.2, len(important_chunks) * 0.05)
            avg_similarity = min(1.0, avg_similarity + boost)
        
        # Boost for code examples (if present)
        code_chunks = [c for c in chunks if c["metadata"].get("has_code", False)]
        if code_chunks:
            avg_similarity = min(1.0, avg_similarity + 0.05)
        
        return round(avg_similarity, 3)

    def _extract_sources(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract source information from chunks."""
        sources = []
        seen_docs = set()
        
        for chunk in chunks:
            doc_name = chunk["metadata"].get("document_name", "Unknown")
            
            if doc_name not in seen_docs:
                seen_docs.add(doc_name)
                sources.append({
                    "document": doc_name,
                    "content_type": chunk["metadata"].get("content_type", "general"),
                    "has_code": chunk["metadata"].get("has_code", False)
                })
        
        return sources

    def _handle_no_context(self, query: str) -> Dict[str, Any]:
        """Handle queries with no relevant context."""
        logger.info(f"No relevant context found for: {query}")
        return {
            "answer": "I couldn't find relevant information in the documentation to answer your question. "
                     "Please try rephrasing your question or ensure the relevant documents have been uploaded.",
            "sources": [],
            "confidence": 0.0,
            "chunks_used": 0,
            "model": "none"
        }


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------
def create_rag_engine(collection_name: str = "programming_docs") -> RAGEngine:
    """
    Create and initialize a RAG engine instance.
    
    Args:
        collection_name: Name for the ChromaDB collection
        
    Returns:
        Initialized RAG engine
    """
    return RAGEngine(collection_name=collection_name)


# ---------------------------------------------------------------------------
# Test/Demo Code
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("="*60)
    print("Testing RAG Engine")
    print("="*60)
    
    # Initialize engine
    engine = create_rag_engine()
    
    # Get stats
    stats = engine.get_collection_stats()
    print(f"\n📊 Collection Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test query if we have data
    if stats.get("total_chunks", 0) > 0:
        test_query = "How do I create a FastAPI endpoint?"
        print(f"\n🔍 Test Query: {test_query}")
        
        try:
            result = engine.query(test_query)
            print(f"\n✨ Answer:")
            print(result["answer"][:300] + "...")
            print(f"\n📚 Sources: {len(result['sources'])} documents")
            print(f"🎯 Confidence: {result['confidence']}")
            print(f"⏱️  Time: {result['total_time_ms']}ms")
        except Exception as e:
            print(f"\n❌ Error: {e}")
    else:
        print("\n⚠️  No documents in collection. Upload documents first!")
    
    print("\n✅ RAG Engine test completed!")