import streamlit as st
import os
from dotenv import load_dotenv
from utils import extract_text, detect_language, get_language_name
from rag_pipeline import RAGPipeline
from llm_handler import GroqHandler

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="South Indian Multilingual QA Chatbot",
    page_icon="🌏",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = None
if "groq_handler" not in st.session_state:
    st.session_state.groq_handler = None
if "document_language" not in st.session_state:
    st.session_state.document_language = None
if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False

# Header
st.title("🌏 South Indian Multilingual Document QA Chatbot")
st.markdown("Upload documents in **Malayalam, Tamil, Telugu, Kannada, or Tulu** and ask questions!")

# Sidebar for file upload
with st.sidebar:
    st.header("📄 Document Upload")
    
    uploaded_file = st.file_uploader(
        "Upload your document",
        type=["pdf", "docx", "txt"],
        help="Supported formats: PDF, DOCX, TXT"
    )
    
    if uploaded_file is not None:
        with st.spinner("Processing document..."):
            try:
                # Extract text from uploaded file
                text = extract_text(uploaded_file)
                
                if not text or len(text.strip()) < 20:
                    st.error("❌ Could not extract sufficient text from the document.")
                else:
                    # Detect language
                    lang_code = detect_language(text)
                    lang_name = get_language_name(lang_code)
                    
                    st.session_state.document_language = lang_code
                    
                    # Display detected language
                    st.success(f"✅ **Detected Language:** {lang_name}")
                    st.info(f"📝 **Document:** {uploaded_file.name}")
                    st.info(f"📊 **Text Length:** {len(text)} characters")
                    
                    # Initialize RAG pipeline if not already done
                    if st.session_state.rag_pipeline is None:
                        st.session_state.rag_pipeline = RAGPipeline()
                    
                    # Initialize Groq handler if not already done
                    if st.session_state.groq_handler is None:
                        st.session_state.groq_handler = GroqHandler()
                    
                    # Process and store document
                    with st.spinner("Embedding and storing document..."):
                        st.session_state.rag_pipeline.process_document(
                            text=text,
                            filename=uploaded_file.name,
                            language=lang_code
                        )
                    
                    st.session_state.document_uploaded = True
                    st.success("✅ Document processed and stored successfully!")
                    
            except Exception as e:
                st.error(f"❌ Error processing document: {str(e)}")
    
    # Display current status
    st.markdown("---")
    st.subheader("📊 Status")
    if st.session_state.document_uploaded:
        st.write(f"✅ Document loaded")
        st.write(f"🌐 Language: {get_language_name(st.session_state.document_language)}")
    else:
        st.write("⏳ No document uploaded")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Main chat interface
if not st.session_state.document_uploaded:
    st.info("👆 Please upload a document from the sidebar to begin!")
else:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about the document..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Retrieve relevant context
                    context_chunks = st.session_state.rag_pipeline.retrieve_context(
                        query=prompt,
                        top_k=5
                    )
                    
                    # Generate answer using Groq
                    answer = st.session_state.groq_handler.generate_answer(
                        query=prompt,
                        context_chunks=context_chunks,
                        language=st.session_state.document_language
                    )
                    
                    st.markdown(answer)
                    
                    # Add assistant response to chat
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                except Exception as e:
                    error_msg = f"❌ Error generating response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Footer
st.markdown("---")
st.markdown(
    "Built with ❤️ using Streamlit, Qdrant, and Groq | "
    "Supports Malayalam, Tamil, Telugu, Kannada, and Tulu"
)