"""
DevDocs AI - Streamlit Frontend
--------------------------------
Professional UI for the Knowledge-Base Search Engine.
"""

import streamlit as st
import requests
import json
import time
from typing import Dict, Any, List, Optional
from datetime import datetime

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_BASE_URL = "http://localhost:8000"

# Page config
st.set_page_config(
    page_title="DevDocs AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .source-card {
        background: #f8f9ff;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    .confidence-high {
        color: #22c55e;
        font-weight: bold;
    }
    .confidence-medium {
        color: #eab308;
        font-weight: bold;
    }
    .confidence-low {
        color: #ef4444;
        font-weight: bold;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem;
        font-size: 1.1rem;
        border-radius: 8px;
        font-weight: bold;
    }
    .stButton>button:hover {
        opacity: 0.9;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def check_api_health() -> Dict[str, Any]:
    """Check if API is available."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.json() if response.status_code == 200 else {}
    except:
        return {}


def upload_document(file, document_name: Optional[str] = None) -> Dict[str, Any]:
    """Upload document to API."""
    try:
        # Use file.getvalue() for the raw content, and file.name for filename/type
        files = {"file": (file.name, file.getvalue(), file.type)}
        data = {}
        if document_name:
            data["document_name"] = document_name
        
        response = requests.post(
            f"{API_BASE_URL}/upload",
            files=files,
            data=data,
            timeout=300
        )
        
        if response.status_code == 201:
            return {"success": True, "data": response.json()}
        else:
            # Try to extract error message from response
            error_detail = response.json().get("detail", response.text)
            return {"success": False, "error": error_detail}
    except Exception as e:
        return {"success": False, "error": str(e)}


def query_knowledge_base(question: str, filters: Optional[Dict] = None) -> Dict[str, Any]:
    """Query the knowledge base."""
    try:
        payload = {"question": question}
        if filters:
            payload["filters"] = filters
        
        response = requests.post(
            f"{API_BASE_URL}/query",
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        else:
            return {"success": False, "error": response.json().get("detail", "Query failed")}
    except Exception as e:
        return {"success": False, "error": str(e)}


def get_examples() -> List[str]:
    """Get example queries."""
    try:
        response = requests.get(f"{API_BASE_URL}/examples", timeout=5)
        if response.status_code == 200:
            # Note: Checking for the correct key from API response
            return response.json().get("examples", [])
    except:
        pass
    return [
        "How do I create a FastAPI endpoint?",
        "What's the difference between async and sync functions?",
        "Show me how to handle database connections"
    ]


def get_stats() -> Dict[str, Any]:
    """Get system statistics."""
    try:
        response = requests.get(f"{API_BASE_URL}/stats", timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return {}


def format_confidence(confidence: float) -> str:
    """Format confidence score with color."""
    if confidence >= 0.7:
        return f'<span class="confidence-high">🟢 {confidence:.1%} (High)</span>'
    elif confidence >= 0.4:
        return f'<span class="confidence-medium">🟡 {confidence:.1%} (Medium)</span>'
    else:
        return f'<span class="confidence-low">🔴 {confidence:.1%} (Low)</span>'

# --- UPDATED CALLBACK FUNCTION FOR ROBUST STATE HANDLING ---
def handle_query_search():
    """Execute query and update session state immediately."""
    query = st.session_state.query_input
    filters = st.session_state.current_filters # Retrieve filters stored in state

    # Reset result and error states at the start of the interaction
    st.session_state.error_message = None
    st.session_state.last_result = None 

    if not query.strip():
        # Set error message in state, which the main loop will display
        st.session_state.error_message = "Please enter a question to search."
        return

    # Use a placeholder/spinner managed within the callback function
    with st.spinner("🔍 Searching documentation..."):
        result = query_knowledge_base(query, filters)
    
    if result["success"]:
        data = result["data"]
        st.session_state.last_result = data
        st.session_state.query_history.insert(0, {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "question": query,
            "confidence": data.get("confidence", 0),
            "chunks_used": data.get("chunks_used", 0)
        })
        # Keep only last 20
        st.session_state.query_history = st.session_state.query_history[:20]
    else:
        # Set API error message in state
        st.session_state.error_message = f"❌ Query failed: {result['error']}"


# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
if "query_history" not in st.session_state:
    st.session_state.query_history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None
# Initialize state for filters to be accessed by the callback
if "current_filters" not in st.session_state:
    st.session_state.current_filters = None
# Initialize state for error messages
if "error_message" not in st.session_state:
    st.session_state.error_message = None


# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------
def main():
    # Header
    st.markdown('<div class="main-header">📚 DevDocs AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Intelligent Programming Documentation Search Engine</div>',
        unsafe_allow_html=True
    )
    
    # Check API health
    health = check_api_health()
    
    if not health:
        st.error("🔴 **API Server is not running!**")
        st.info("Please start the API server: `python src/api.py`")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.title("⚙️ Control Panel")
        
        # API Status
        st.subheader("📊 System Status")
        if health.get("status") == "healthy":
            st.success("🟢 API: Operational")
        else:
            st.warning("🟡 API: Degraded")
        
        stats = health.get("collection_stats", {})
        st.metric("Chunks Indexed", stats.get("total_chunks", 0))
        
        st.divider()
        
        # Document Upload Section (MODIFIED FOR MULTIPLE FILES)
        st.subheader("📁 Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose files to upload",
            type=["pdf", "txt"],
            accept_multiple_files=True, # Key change for multiple files
            help="Upload one or more PDF or TXT documents"
        )
        
        custom_name = st.text_input(
            "Custom Document Name (Optional, applies to first file only)",
            placeholder="e.g., FastAPI Tutorial v1"
        )
        
        if st.button("📤 Upload & Process", key="upload_btn"):
            if uploaded_files:
                # Clear existing search results and errors upon starting a new upload process
                st.session_state.last_result = None
                st.session_state.error_message = None
                
                with st.spinner(f"Processing {len(uploaded_files)} documents..."):
                    total_chunks = 0
                    
                    for i, uploaded_file in enumerate(uploaded_files):
                        # Use custom name for the first file, or the original name for the rest
                        doc_name = custom_name if i == 0 and custom_name else uploaded_file.name
                        
                        # Call API for each file
                        result = upload_document(uploaded_file, document_name=doc_name)
                        
                        if result["success"]:
                            data = result["data"]
                            total_chunks += data['chunks_processed']
                            st.success(f"✅ **{data['document_name']}** processed! ({data['chunks_processed']} chunks)")
                        else:
                            st.error(f"❌ Upload failed for {uploaded_file.name}: {result['error']}")
                    
                    if total_chunks > 0:
                         st.info(f"📊 Total of {total_chunks} chunks created.")
                    
                    # Refresh UI after processing all files
                    time.sleep(1)
                    st.rerun()
            else:
                st.warning("Please select one or more files first")
        
        st.divider()
        
        # Query Filters
        st.subheader("🔍 Query Filters")
        use_filters = st.checkbox("Enable Filters", value=False)
        
        filters = None
        if use_filters:
            content_type = st.selectbox(
                "Content Type",
                ["", "api_documentation", "code_example", "setup_guide", "general_documentation"]
            )
            importance = st.selectbox(
                "Importance",
                ["", "high", "medium"]
            )
            
            filters = {}
            if content_type:
                filters["content_type"] = content_type
            if importance:
                filters["importance"] = importance
        
        # Store filters in state so they can be accessed by the callback function
        st.session_state.current_filters = filters

        st.divider()
        
        # Statistics
        if st.button("📊 Refresh Stats"):
            st.rerun()
        
        system_stats = get_stats()
        if system_stats:
            kb_stats = system_stats.get("knowledge_base", {})
            with st.expander("📈 Detailed Stats", expanded=False):
                st.json(kb_stats)
    
    # Main Content Area
    tab1, tab2, tab3 = st.tabs(["🔍 Search", "📜 History", "ℹ️ About"])
    
    # -------------------------------------------------------------------------
    # Tab 1: Search
    # -------------------------------------------------------------------------
    with tab1:
        st.subheader("Ask Your Question")
        
        # Example queries
        examples = get_examples()
        example_cols = st.columns(len(examples))
        
        # Ensure correct key name is used when setting state for example buttons
        for idx, (col, example) in enumerate(zip(example_cols, examples)):
            with col:
                if st.button(f"💡 {example[:30]}...", key=f"example_{idx}"):
                    st.session_state.query_input = example
        
        # Query input
        st.text_area(
            "Your Question:",
            height=100,
            placeholder="e.g., How do I create an async FastAPI endpoint with path parameters?",
            key="query_input" # Input uses session state key
        )
        
        col1, col2 = st.columns([3, 1])
        with col1:
            # --- MODIFIED BUTTON USES on_click CALLBACK ---
            st.button(
                "🚀 Search Knowledge Base", 
                type="primary", 
                use_container_width=True, 
                on_click=handle_query_search
            )
        with col2:
            clear_button = st.button("🗑️ Clear", use_container_width=True)
        
        if clear_button:
            st.session_state.query_input = ""
            st.session_state.last_result = None
            st.session_state.error_message = None # Clear any pending error
            st.rerun()
        
        # --- DISPLAY ERROR IF PRESENT ---
        if st.session_state.error_message:
            st.error(st.session_state.error_message)
        
        # The main logic to display results now relies entirely on session state
        if st.session_state.last_result:
            st.divider()
            data = st.session_state.last_result
            
            # Metrics row
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("⏱️ Response Time", f"{data.get('processing_time_ms', 0):.0f}ms")
            with col2:
                st.metric("📊 Chunks Used", data.get("chunks_used", 0))
            with col3:
                confidence = data.get("confidence", 0)
                st.markdown(f"**🎯 Confidence**")
                st.markdown(format_confidence(confidence), unsafe_allow_html=True)
            with col4:
                st.metric("📚 Sources", len(data.get("sources", [])))
            
            st.divider()
            
            # Answer
            st.subheader("✨ Answer")
            st.markdown(data.get("answer", "No answer provided"))
            
            # Sources
            st.divider()
            st.subheader("📚 Sources")
            
            sources = data.get("sources", [])
            if sources:
                for idx, source in enumerate(sources, 1):
                    with st.container():
                        st.markdown(f"""
                        <div class="source-card">
                            <strong>📄 Source {idx}: {source.get('document', 'Unknown')}</strong><br>
                            <small>
                                Type: {source.get('content_type', 'unknown')} | 
                                Has Code: {'✅ Yes' if source.get('has_code') else '❌ No'}
                            </small>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("No sources available")
            
            # Copy button
            st.subheader("📋 Answer for Copying")
            st.code(data.get("answer", ""), language="markdown")
    
    # -------------------------------------------------------------------------
    # Tab 2: History
    # -------------------------------------------------------------------------
    with tab2:
        st.subheader("📜 Query History")
        
        if st.session_state.query_history:
            for idx, entry in enumerate(st.session_state.query_history):
                with st.expander(
                    f"🕐 {entry['timestamp']} - {entry['question'][:60]}...",
                    expanded=(idx == 0)
                ):
                    st.write(f"**Question:** {entry['question']}")
                    st.write(f"**Confidence:** {entry['confidence']:.1%}")
                    st.write(f"**Chunks Used:** {entry['chunks_used']}")
        else:
            st.info("No query history yet. Start searching!")
        
        if st.session_state.query_history:
            if st.button("🗑️ Clear History"):
                st.session_state.query_history = []
                st.rerun()
    
    # -------------------------------------------------------------------------
    # Tab 3: About
    # -------------------------------------------------------------------------
    with tab3:
        st.subheader("ℹ️ About DevDocs AI")
        
        st.markdown("""
        ### 🎯 What is DevDocs AI?
        
        DevDocs AI is an intelligent programming documentation search engine that uses:
        - **Semantic Search**: Find information by meaning, not just keywords
        - **RAG (Retrieval-Augmented Generation)**: Combines search with AI generation
        - **Metadata-Based Re-ranking**: Smart boosting based on content type and importance
        
        ### 🚀 Key Features
        
        - 📄 **Multi-format Support**: PDF and TXT documents
        - 🔍 **Smart Search**: Understanding context and intent
        - 🎯 **Confidence Scoring**: Know how reliable answers are
        - 📚 **Source Attribution**: Always cites where information comes from
        - ⚡ **Fast Responses**: Optimized for speed
        
        ### 🛠️ Technology Stack
        
        - **Backend**: FastAPI + Python
        - **Frontend**: Streamlit
        - **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
        - **Vector DB**: ChromaDB
        - **LLM**: Groq (Llama 3.1)
        - **Document Processing**: LangChain
        
        ### 📊 Current System Stats
        """)
        
        system_stats = get_stats()
        if system_stats:
            st.json(system_stats)
        
        st.markdown("""
        ### 👨‍💻 Usage Tips
        
        1. **Upload Documents**: Start by uploading programming documentation
        2. **Ask Natural Questions**: Write questions as you would ask a colleague
        3. **Use Filters**: Narrow down to specific content types
        4. **Check Confidence**: Higher confidence = more reliable answer
        5. **Verify Sources**: Always check the cited sources
        
        ---
        
        **Version**: 1.0.0 | **Built with** ❤️ **for developers**
        """)


# ---------------------------------------------------------------------------
# Run App
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()