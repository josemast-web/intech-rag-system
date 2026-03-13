import streamlit as st
from src.config import Config, logger
from src.drive_loader import IntechDriveLoader
from src.vector_service import IntechVectorService

# UI Configuration
st.set_page_config(page_title="Intech Engineering - Technical RAG", layout="wide")

def initialize_system():
    """Initializes the backend services and stores them in session state."""
    if 'vector_service' not in st.session_state:
        st.session_state.vector_service = IntechVectorService()
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

def run_indexing():
    """Triggers the document ingestion and indexing process."""
    with st.spinner("Accessing Google Drive and updating vector index..."):
        try:
            loader = IntechDriveLoader()
            documents = loader.load_documents()
            st.session_state.vector_service.build_or_load_index(documents)
            st.success("Index updated successfully. System ready for queries.")
            logger.info("Manual index refresh completed by user.")
        except Exception as e:
            st.error(f"Indexing failed: {str(e)}")
            logger.error(f"User-triggered indexing failed: {str(e)}")

# Initialize App
initialize_system()

# Sidebar: Management and Status
with st.sidebar:
    st.title("System Control")
    st.info("Target Folder: Intech Engineering (Shared)")
    
    if st.button("🔄 Refresh Knowledge Base"):
        run_indexing()
    
    st.divider()
    st.markdown("### Excluded Files")
    for item in Config.EXCLUDED_FILES:
        st.write(f"- {item}")

# Main Chat Interface
st.title("Intech Technical Knowledge Assistant")
st.markdown("Ask questions about Test Stands, technical specs, or QuickBooks reports.")

# Display Chat History
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Query Input
if prompt := st.chat_input("Enter your technical inquiry here..."):
    # Add user message to history
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing documentation..."):
            response = st.session_state.vector_service.execute_query(prompt)
            st.markdown(response)
    
    # Add assistant response to history
    st.session_state.chat_history.append({"role": "assistant", "content": str(response)})
