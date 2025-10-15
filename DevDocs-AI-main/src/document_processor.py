"""
Document Processor Module
-------------------------
Handles document ingestion and preprocessing for the RAG system.
Uses LangChain for robust document loading and chunking.
"""

import os
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Add parent directory to path for config import
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CHUNK_SIZE, CHUNK_OVERLAP, SUPPORTED_EXTENSIONS

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
# Custom Exception
# ---------------------------------------------------------------------------
class DocumentProcessingError(Exception):
    """Raised when document processing fails."""
    pass


# ---------------------------------------------------------------------------
# Document Processor
# ---------------------------------------------------------------------------
class DocumentProcessor:
    """
    Processes documents using LangChain with custom metadata enrichment
    optimized for technical documentation.
    """

    def __init__(self) -> None:
        """Initialize the document processor with text splitter."""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        )
        self.supported_extensions = SUPPORTED_EXTENSIONS
        logger.info("Document processor initialized")

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------
    def process_file(self, file_path: str, document_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Process a single document file.
        
        Args:
            file_path: Path to the document file
            document_name: Optional custom name for the document
            
        Returns:
            List of processed chunks with metadata
        """
        document_name = document_name or os.path.basename(file_path)
        logger.info(f"Processing document: {document_name}")

        try:
            # Load document
            raw_documents = self._load_document(file_path)
            
            # Add metadata
            enriched_docs = self._add_metadata(raw_documents, document_name)
            
            # Split into chunks
            chunks = self.text_splitter.split_documents(enriched_docs)
            
            # Enhance chunks
            enhanced_chunks = self._enhance_chunks(chunks)
            
            # Standardize output format
            standardized_chunks = self._standardize_chunks(enhanced_chunks, document_name)

            logger.info(f"Successfully processed {document_name}: {len(standardized_chunks)} chunks")
            return standardized_chunks

        except Exception as e:
            logger.exception(f"Failed to process {document_name}: {e}")
            raise DocumentProcessingError(f"Processing failed: {str(e)}") from e

    def process_multiple_files(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Process multiple document files.
        
        Args:
            file_paths: List of file paths to process
            
        Returns:
            Combined list of all processed chunks
        """
        all_chunks = []
        failed_files = []
        
        logger.info(f"Processing {len(file_paths)} files...")
        
        for file_path in file_paths:
            try:
                chunks = self.process_file(file_path)
                all_chunks.extend(chunks)
            except DocumentProcessingError as e:
                logger.warning(f"Skipping {file_path}: {e}")
                failed_files.append(file_path)
        
        if failed_files:
            logger.warning(f"Failed to process {len(failed_files)} files")
        
        logger.info(f"Successfully processed {len(file_paths) - len(failed_files)} files")
        return all_chunks

    # -----------------------------------------------------------------------
    # Internal Methods
    # -----------------------------------------------------------------------
    def _load_document(self, file_path: str) -> List[Document]:
        """Load document using appropriate LangChain loader."""
        file_extension = os.path.splitext(file_path)[1].lower()

        try:
            if file_extension == ".pdf":
                loader = PyPDFLoader(file_path)
            elif file_extension == ".txt":
                loader = TextLoader(file_path, encoding="utf-8")
            else:
                raise DocumentProcessingError(
                    f"Unsupported file type: {file_extension}. "
                    f"Supported: {', '.join(self.supported_extensions)}"
                )

            documents = loader.load()
            
            if not documents:
                raise DocumentProcessingError("No content extracted from document")
                
            return documents

        except Exception as e:
            logger.error(f"Document loading failed for {file_path}: {e}")
            raise DocumentProcessingError(f"Failed to load document: {str(e)}") from e

    def _add_metadata(self, documents: List[Document], document_name: str) -> List[Document]:
        """Add rich metadata to documents."""
        for doc in documents:
            # Preserve existing metadata
            doc.metadata = {**doc.metadata}

            # Add processing metadata
            doc.metadata.update({
                "document_name": document_name,
                "processed_time": datetime.now().isoformat(),
                "processor_version": "1.0.0",
            })

            # Detect content type based on content analysis
            preview = doc.page_content[:300].lower()
            
            if any(keyword in preview for keyword in ["api", "endpoint", "route", "request", "response"]):
                doc.metadata["content_type"] = "api_documentation"
            elif any(keyword in preview for keyword in ["def ", "class ", "function", "import ", "const ", "let ", "var "]):
                doc.metadata["content_type"] = "code_example"
            elif any(keyword in preview for keyword in ["install", "setup", "configuration", "getting started"]):
                doc.metadata["content_type"] = "setup_guide"
            else:
                doc.metadata["content_type"] = "general_documentation"

        return documents

    def _enhance_chunks(self, chunks: List[Document]) -> List[Document]:
        """Add programming-specific enhancements to chunks."""
        enhanced_chunks = []

        for idx, chunk in enumerate(chunks):
            content = chunk.page_content
            
            # Ensure metadata isolation
            chunk.metadata = {**chunk.metadata}

            # Detect code snippets
            has_code = self._contains_code(content)
            chunk.metadata["has_code"] = has_code
            
            if has_code:
                chunk.metadata["code_language"] = self._detect_language(content)

            # Calculate importance score
            importance_score = self._calculate_importance(content)
            chunk.metadata["importance"] = "high" if importance_score > 0.7 else "medium"
            chunk.metadata["importance_score"] = importance_score

            # Add sequence information
            chunk.metadata["chunk_sequence"] = idx

            enhanced_chunks.append(chunk)

        return enhanced_chunks

    def _standardize_chunks(self, chunks: List[Document], document_name: str) -> List[Dict[str, Any]]:
        """Convert chunks to standardized dictionary format."""
        standardized = []
        
        for idx, chunk in enumerate(chunks):
            # Generate unique chunk ID
            chunk_id = f"{document_name}_chunk_{idx}"
            
            standardized.append({
                "chunk_id": chunk_id,
                "content": chunk.page_content,
                "metadata": chunk.metadata,
            })
        
        return standardized

    # -----------------------------------------------------------------------
    # Utility Methods
    # -----------------------------------------------------------------------
    def _contains_code(self, text: str) -> bool:
        """Detect if text contains code snippets."""
        code_indicators = [
            "def ", "class ", "function ", "import ", "export ",
            "const ", "let ", "var ", "```", "public ", "private ",
            "return ", "async ", "await "
        ]
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in code_indicators)

    def _detect_language(self, text: str) -> str:
        """Detect programming language in text."""
        text_lower = text.lower()
        
        # Python indicators
        if any(keyword in text_lower for keyword in ["def ", "import ", "__init__", "self."]):
            return "python"
        
        # JavaScript/TypeScript indicators
        if any(keyword in text_lower for keyword in ["const ", "let ", "function ", "=>", "async "]):
            return "javascript"
        
        # Java indicators
        if any(keyword in text_lower for keyword in ["public class", "void ", "static ", "private "]):
            return "java"
        
        return "unknown"

    def _calculate_importance(self, text: str) -> float:
        """Calculate importance score based on content."""
        text_lower = text.lower()
        score = 0.5  # Base score
        
        # Important keywords boost score
        important_keywords = [
            "warning", "important", "note:", "caution", "deprecated",
            "security", "critical", "error", "required", "must"
        ]
        
        # Count important keywords
        keyword_count = sum(1 for keyword in important_keywords if keyword in text_lower)
        
        # Boost score based on keyword density
        if keyword_count > 0:
            score += min(0.3, keyword_count * 0.1)
        
        # Has code examples (valuable)
        if self._contains_code(text):
            score += 0.2
        
        return min(1.0, score)

    def get_stats(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get processing statistics."""
        if not chunks:
            return {
                "total_chunks": 0,
                "total_documents": 0,
            }
        
        # Calculate statistics
        total_chars = sum(len(c["content"]) for c in chunks)
        code_chunks = sum(1 for c in chunks if c["metadata"].get("has_code", False))
        important_chunks = sum(1 for c in chunks if c["metadata"].get("importance") == "high")
        
        # Count unique documents
        documents = set(c["metadata"].get("document_name") for c in chunks)
        
        # Content type distribution
        content_types = {}
        for chunk in chunks:
            ct = chunk["metadata"].get("content_type", "unknown")
            content_types[ct] = content_types.get(ct, 0) + 1
        
        return {
            "total_chunks": len(chunks),
            "total_documents": len(documents),
            "total_characters": total_chars,
            "avg_chunk_size": round(total_chars / len(chunks), 2),
            "code_chunks": code_chunks,
            "important_chunks": important_chunks,
            "content_types": content_types,
            "documents": list(documents)
        }


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------
def process_document(file_path: str, document_name: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Convenience function to process a single document.
    
    Args:
        file_path: Path to document file
        document_name: Optional custom document name
        
    Returns:
        List of processed chunks
    """
    processor = DocumentProcessor()
    return processor.process_file(file_path, document_name)


# ---------------------------------------------------------------------------
# Test/Demo Code
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("="*60)
    print("Testing Document Processor")
    print("="*60)
    
    processor = DocumentProcessor()
    
    # Check for sample documents
    from config import RAW_DOCS_DIR
    pdf_files = list(Path(RAW_DOCS_DIR).glob("*.pdf"))
    txt_files = list(Path(RAW_DOCS_DIR).glob("*.txt"))
    
    all_files = pdf_files + txt_files
    
    if not all_files:
        print(f"\n⚠️  No documents found in {RAW_DOCS_DIR}")
        print("Please add PDF or TXT files to test the processor.")
    else:
        print(f"\n📚 Found {len(all_files)} documents")
        
        # Process first file as example
        test_file = all_files[0]
        print(f"\n🧪 Processing: {test_file.name}")
        
        try:
            chunks = processor.process_file(str(test_file))
            stats = processor.get_stats(chunks)
            
            print("\n📊 Processing Statistics:")
            for key, value in stats.items():
                if key != 'documents':
                    print(f"  {key}: {value}")
            
            print(f"\n📝 Sample Chunk:")
            sample = chunks[0]
            print(f"  ID: {sample['chunk_id']}")
            print(f"  Content Type: {sample['metadata']['content_type']}")
            print(f"  Has Code: {sample['metadata']['has_code']}")
            print(f"  Importance: {sample['metadata']['importance']}")
            print(f"  Preview: {sample['content'][:200]}...")
            
            print("\n✅ Document processor test completed successfully!")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")