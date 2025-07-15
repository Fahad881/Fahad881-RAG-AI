# ===================================================================
# CLEAN RAG APPLICATION - WHITE THEME
# Focus on core functionality: Upload documents and chat with LLM
# ===================================================================

# IMPORT ALL REQUIRED LIBRARIES
import streamlit as st
import pdfplumber
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
import fitz
import io
import os
import pandas as pd
import arabic_reshaper
from bidi.algorithm import get_display
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import tempfile
import shutil
from typing import List, Optional, Tuple
import logging

# APPLICATION CONFIGURATION
class Config:
    MAX_FILE_SIZE_MB = 50
    CHUNK_SIZE = 2000
    CHUNK_OVERLAP = 200
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_MODEL = "llama3-70b-8192"
    GROQ_BASE_URL = "https://api.groq.com/openai/v1"
    RETRIEVAL_K = 3

# CREATE DIRECTORY STRUCTURE
TEXT_DIR = "extracted_texts"
CHUNK_DIR = "output_chunks"
TABLE_DIR = "extracted_tables"
VECTOR_DIR = "vectorstore/faiss_index"

for dir_path in [TEXT_DIR, CHUNK_DIR, TABLE_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# STREAMLIT PAGE CONFIGURATION
st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CLEAN WHITE THEME CSS
st.markdown("""
<style>
    .stApp {
        background-color: #ffffff;
        color: #333333;
    }
    
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background-color: #ffffff;
    }
    
    h1, h2, h3 {
        color: #2c3e50;
        font-weight: 600;
    }
    
    .css-1d391kg {
        background-color: #f8f9fa;
        border-right: 1px solid #e9ecef;
    }
    
    .stButton > button {
        background-color: #007bff;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: background-color 0.3s;
    }
    
    .stButton > button:hover {
        background-color: #0056b3;
    }
    
    .stFileUploader {
        border: 2px dashed #007bff;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background-color: #f8f9fa;
    }
    
    .stSuccess {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
    }
    
    .stError {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
    }
    
    .stWarning {
        background-color: #fff3cd;
        color: #856404;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
    }
    
    .chat-user {
        background-color: #007bff;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        margin-left: 20%;
    }
    
    .chat-assistant {
        background-color: #f8f9fa;
        color: #333;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        margin-right: 20%;
        border: 1px solid #e9ecef;
    }
    
    .stTextInput input, .stTextArea textarea {
        border: 1px solid #ced4da;
        border-radius: 5px;
        padding: 0.5rem;
    }
    
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: #007bff;
        box-shadow: 0 0 0 0.2rem rgba(0, 123, 255, 0.25);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 5px 5px 0 0;
        padding: 0.5rem 1rem;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #007bff;
        color: white;
        border-color: #007bff;
    }
    
    .stProgress .st-bo {
        background-color: #007bff;
    }
    
    .metric-container {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #007bff;
    }
    
    .metric-label {
        color: #6c757d;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# SESSION STATE INITIALIZATION
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'vector_store_ready' not in st.session_state:
    st.session_state.vector_store_ready = False

# SIDEBAR CONFIGURATION
with st.sidebar:
    st.header("⚙️ Configuration")
    
    api_key = st.text_input(
        "🔑 Groq API Key",
        type="password",
        help="Enter your Groq API key from https://console.groq.com"
    )
    
    with st.expander("Advanced Settings"):
        chunk_size = st.slider("Chunk Size", 500, 4000, Config.CHUNK_SIZE)
        chunk_overlap = st.slider("Chunk Overlap", 50, 500, Config.CHUNK_OVERLAP)
        retrieval_k = st.slider("Retrieval K", 1, 10, Config.RETRIEVAL_K)
    
    st.header("📊 System Status")
    
    col1, col2 = st.columns(2)
    with col1:
        files_count = len(st.session_state.processed_files)
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{files_count}</div>
            <div class="metric-label">Files Processed</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        status = "✅ Ready" if st.session_state.vector_store_ready else "❌ Not Ready"
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Vector Store</div>
            <div style="color: {'#28a745' if st.session_state.vector_store_ready else '#dc3545'}; font-weight: bold;">
                {status}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    if st.button("🗑️ Clear All Data", type="secondary"):
        if st.session_state.get('confirm_clear', False):
            clear_all_data()
            st.session_state.confirm_clear = False
            st.rerun()
        else:
            st.session_state.confirm_clear = True
            st.warning("Click again to confirm deletion")

# MAIN HEADER
st.title("🤖 RAG Assistant")
st.markdown("Upload documents, ask questions, and get AI-powered answers with source citations.")

# UTILITY FUNCTIONS

def clear_all_data():
    """Clear all processed data and reset application state"""
    for folder in [TEXT_DIR, CHUNK_DIR, VECTOR_DIR]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)
    
    st.session_state.processed_files = []
    st.session_state.chat_history = []
    st.session_state.vector_store_ready = False
    st.success("✅ All data cleared successfully")

def validate_file(file_bytes: bytes, filename: str) -> bool:
    """Validate uploaded file"""
    if len(file_bytes) > Config.MAX_FILE_SIZE_MB * 1024 * 1024:
        st.error(f"❌ File '{filename}' is too large (max {Config.MAX_FILE_SIZE_MB}MB)")
        return False
    
    if len(file_bytes) == 0:
        st.error(f"❌ File '{filename}' is empty")
        return False
    
    return True

def detect_text_pdf(file_bytes: bytes) -> bool:
    """Check if PDF contains extractable text"""
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages[:3]:
                text = page.extract_text()
                if text and text.strip():
                    return True
    except:
        pass
    return False

def extract_text_from_pdf(file_bytes: bytes, filename: str) -> str:
    """Extract text from PDF using the best available method"""
    if not validate_file(file_bytes, filename):
        return ""
    
    try:
        if detect_text_pdf(file_bytes):
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                if text.strip():
                    return text.strip()
        
        st.info(f"📄 Using OCR for '{filename}' (scanned document detected)")
        return extract_text_with_ocr(file_bytes)
        
    except Exception as e:
        st.error(f"❌ Failed to extract text from '{filename}': {str(e)}")
        return ""

def extract_text_with_ocr(file_bytes: bytes) -> str:
    """Extract text using OCR"""
    try:
        images = convert_from_bytes(file_bytes)
        text_parts = []
        
        for i, image in enumerate(images):
            try:
                page_text = pytesseract.image_to_string(image, lang='ara+eng')
                if page_text.strip():
                    text_parts.append(page_text)
            except Exception as e:
                st.warning(f"⚠️ OCR failed for page {i+1}: {str(e)}")
                continue
        
        return "\n".join(text_parts)
        
    except Exception as e:
        st.error(f"❌ OCR processing failed: {str(e)}")
        return ""

def reshape_arabic(text: str) -> str:
    """Fix Arabic text display issues"""
    if isinstance(text, str) and any('\u0600' <= c <= '\u06FF' for c in text):
        try:
            reshaped = arabic_reshaper.reshape(text)
            return get_display(reshaped)
        except:
            return text
    return text

def extract_tables(file_bytes: bytes, filename: str) -> Optional[pd.DataFrame]:
    """Extract tables from PDF documents"""
    try:
        combined_rows = []
        max_cols = 0
        
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for table in tables:
                    for row in table:
                        cleaned = [reshape_arabic(cell.strip() if cell else "") for cell in row]
                        combined_rows.append(cleaned)
                        max_cols = max(max_cols, len(cleaned))
        
        if combined_rows:
            df = pd.DataFrame([row + [""] * (max_cols - len(row)) for row in combined_rows])
            path = os.path.join(TABLE_DIR, f"{filename}.csv")
            df.to_csv(path, index=False, encoding="utf-8-sig")
            return df
            
    except Exception as e:
        st.warning(f"⚠️ Table extraction failed for '{filename}': {str(e)}")
    
    return None

def extract_images(file_bytes: bytes, filename: str) -> List[Image.Image]:
    """Extract images from PDF documents"""
    images = []
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        
        for page_num, page in enumerate(doc):
            for img_index, img in enumerate(page.get_images(full=True)):
                try:
                    xref = img[0]
                    image_bytes = doc.extract_image(xref)["image"]
                    images.append(Image.open(io.BytesIO(image_bytes)))
                except Exception as e:
                    st.warning(f"⚠️ Failed to extract image {img_index} from page {page_num+1}")
                    continue
        doc.close()
    except Exception as e:
        st.warning(f"⚠️ Image extraction failed for '{filename}': {str(e)}")
    
    return images

def chunk_text_files(chunk_size: int = Config.CHUNK_SIZE, chunk_overlap: int = Config.CHUNK_OVERLAP):
    """Split text files into smaller chunks for processing"""
    try:
        if os.path.exists(CHUNK_DIR):
            shutil.rmtree(CHUNK_DIR)
        os.makedirs(CHUNK_DIR, exist_ok=True)
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        chunk_count = 0
        for filename in os.listdir(TEXT_DIR):
            file_path = os.path.join(TEXT_DIR, filename)
            with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
                raw_text = file.read()
            
            if raw_text.strip():
                documents = splitter.create_documents([raw_text])
                for doc in documents:
                    chunk_count += 1
                    chunk_filename = f"chunk_{chunk_count:04d}.txt"
                    with open(os.path.join(CHUNK_DIR, chunk_filename), "w", encoding="utf-8") as f:
                        f.write(doc.page_content)
        
        return chunk_count
        
    except Exception as e:
        st.error(f"❌ Text chunking failed: {str(e)}")
        return 0

@st.cache_resource
def get_embeddings_model():
    """Load and cache the embeddings model"""
    try:
        return HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)
    except Exception as e:
        st.error(f"❌ Failed to load embeddings model: {str(e)}")
        return None

def embed_and_store_chunks() -> bool:
    """Convert text chunks to embeddings and store in vector database"""
    try:
        documents = []
        chunk_files = sorted([f for f in os.listdir(CHUNK_DIR) if f.endswith('.txt')])
        
        if not chunk_files:
            st.warning("⚠️ No text chunks found to embed")
            return False
        
        for fname in chunk_files:
            file_path = os.path.join(CHUNK_DIR, fname)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    documents.append(Document(page_content=content, metadata={"source": fname}))
        
        if not documents:
            st.warning("⚠️ No valid documents found to embed")
            return False
        
        embeddings = get_embeddings_model()
        if not embeddings:
            return False
        
        vector_db = FAISS.from_documents(documents, embeddings)
        
        if os.path.exists(VECTOR_DIR):
            shutil.rmtree(VECTOR_DIR)
        os.makedirs(VECTOR_DIR, exist_ok=True)
        vector_db.save_local(VECTOR_DIR)
        
        return True
        
    except Exception as e:
        st.error(f"❌ Embedding failed: {str(e)}")
        return False

def process_uploaded_files(uploaded_files, chunk_size: int, chunk_overlap: int):
    """Main processing pipeline for uploaded files"""
    if not uploaded_files:
        return False
    
    for folder in [TEXT_DIR, CHUNK_DIR]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_files = len(uploaded_files)
    processed_files = []
    
    for i, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"Processing {uploaded_file.name}...")
        progress_bar.progress(i / total_files)
        
        try:
            file_bytes = uploaded_file.read()
            name_base = os.path.splitext(uploaded_file.name)[0]
            
            text = extract_text_from_pdf(file_bytes, uploaded_file.name)
            if text:
                text_file_path = os.path.join(TEXT_DIR, uploaded_file.name + ".txt")
                with open(text_file_path, "w", encoding="utf-8") as f:
                    f.write(text)
                
                processed_files.append({
                    'name': uploaded_file.name,
                    'text_length': len(text),
                    'status': 'success'
                })
            else:
                processed_files.append({
                    'name': uploaded_file.name,
                    'text_length': 0,
                    'status': 'failed'
                })
                continue
            
            tables_df = extract_tables(file_bytes, name_base)
            if tables_df is not None:
                st.success(f"✅ Extracted {len(tables_df)} table rows from {uploaded_file.name}")
            
            images = extract_images(file_bytes, uploaded_file.name)
            if images:
                st.success(f"✅ Extracted {len(images)} images from {uploaded_file.name}")
                
        except Exception as e:
            st.error(f"❌ Failed to process {uploaded_file.name}: {str(e)}")
            processed_files.append({
                'name': uploaded_file.name,
                'text_length': 0,
                'status': 'error'
            })
    
    status_text.text("Creating text chunks...")
    progress_bar.progress(0.8)
    chunk_count = chunk_text_files(chunk_size, chunk_overlap)
    
    status_text.text("Creating embeddings...")
    progress_bar.progress(0.9)
    success = embed_and_store_chunks()
    
    progress_bar.progress(1.0)
    status_text.text("✅ Processing complete!")
    
    st.session_state.processed_files = processed_files
    st.session_state.vector_store_ready = success
    
    return success

def answer_query(question: str, api_key: str, retrieval_k: int = Config.RETRIEVAL_K) -> Tuple[str, List]:
    """Answer user question using RAG"""
    if not api_key:
        raise ValueError("API key is required")
    
    if not os.path.exists(VECTOR_DIR) or not os.listdir(VECTOR_DIR):
        raise ValueError("Vector store not found. Please upload and process documents first.")
    
    try:
        embeddings = get_embeddings_model()
        if not embeddings:
            raise ValueError("Failed to load embeddings model")
        
        vector_db = FAISS.load_local(
            VECTOR_DIR,
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
        
        llm = ChatOpenAI(
            base_url=Config.GROQ_BASE_URL,
            api_key=api_key,
            model=Config.LLM_MODEL,
            temperature=0.1
        )
        
        # Build conversation context from chat history
        context = ""
        if st.session_state.chat_history:
            context = "Previous conversation:\n"
            for q, a in st.session_state.chat_history[-3:]:  # Last 3 exchanges
                context += f"Q: {q}\nA: {a}\n\n"
            context += f"Current question: {question}"
        else:
            context = question
        
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        rag_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_db.as_retriever(search_kwargs={"k": retrieval_k}),
            memory=memory,
            return_source_documents=True,
            output_key="answer"
        )
        
        response = rag_chain.invoke({"question": context})
        return response["answer"], response["source_documents"]
        
    except Exception as e:
        raise Exception(f"Query processing failed: {str(e)}")

# MAIN APPLICATION INTERFACE
def main():
    """Main application interface with tabs"""
    tab1, tab2, tab3 = st.tabs(["📁 Upload & Process", "💬 Chat", "📊 Analytics"])
    
    # UPLOAD & PROCESS TAB
    with tab1:
        st.header("📄 Document Upload & Processing")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_files = st.file_uploader(
                "Choose files to upload",
                type=["pdf", "jpg", "jpeg", "png"],
                accept_multiple_files=True,
                help="Upload PDFs, images, or scanned documents"
            )
            
            if uploaded_files:
                st.write(f"📁 Selected {len(uploaded_files)} file(s)")
                for file in uploaded_files:
                    st.write(f"• {file.name} ({file.size:,} bytes)")
        
        with col2:
            st.subheader("Processing Options")
            use_custom_chunking = st.checkbox("Custom chunking settings")
            
            if use_custom_chunking:
                custom_chunk_size = st.number_input("Chunk Size", 500, 4000, chunk_size)
                custom_chunk_overlap = st.number_input("Chunk Overlap", 50, 500, chunk_overlap)
            else:
                custom_chunk_size = chunk_size
                custom_chunk_overlap = chunk_overlap
        
        if st.button("🚀 Process Documents", type="primary", disabled=not uploaded_files):
            if not uploaded_files:
                st.error("Please upload at least one file")
            else:
                success = process_uploaded_files(uploaded_files, custom_chunk_size, custom_chunk_overlap)
                if success:
                    st.success("✅ Documents processed successfully!")
                else:
                    st.error("❌ Processing failed")
        
        if st.session_state.processed_files:
            st.subheader("📋 Processing Results")
            results_df = pd.DataFrame(st.session_state.processed_files)
            st.dataframe(results_df, use_container_width=True)
    
    # CHAT TAB
    with tab2:
        st.header("💬 Chat with Your Documents")
        
        if not api_key:
            st.warning("⚠️ Please enter your Groq API key in the sidebar")
            st.stop()
        
        if not st.session_state.vector_store_ready:
            st.warning("⚠️ Please upload and process documents first")
            st.stop()
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.success("Chat history cleared!")
            st.rerun()
        
        # Display chat history
        for i, (q, a) in enumerate(st.session_state.chat_history):
            st.markdown(f"""
            <div class="chat-user">
                <strong>You:</strong> {q}
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="chat-assistant">
                <strong>Assistant:</strong> {a}
            </div>
            """, unsafe_allow_html=True)
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your documents..."):
            st.session_state.chat_history.append((prompt, ""))
            
            st.markdown(f"""
            <div class="chat-user">
                <strong>You:</strong> {prompt}
            </div>
            """, unsafe_allow_html=True)
            
            try:
                with st.spinner("🤔 Thinking..."):
                    answer, sources = answer_query(prompt, api_key, retrieval_k)
                
                st.markdown(f"""
                <div class="chat-assistant">
                    <strong>Assistant:</strong> {answer}
                </div>
                """, unsafe_allow_html=True)
                
                st.session_state.chat_history[-1] = (prompt, answer)
                
                if sources:
                    with st.expander("📚 Source Documents"):
                        for i, doc in enumerate(sources, 1):
                            st.write(f"**Source {i}:**")
                            preview = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
                            st.write(preview)
                            st.write("---")
                            
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.session_state.chat_history.pop()
    
    # ANALYTICS TAB
    with tab3:
        st.header("📊 Document Analytics")
        
        if not st.session_state.processed_files:
            st.info("No documents processed yet")
            return
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_files = len(st.session_state.processed_files)
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">{total_files}</div>
                <div class="metric-label">Total Files</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            successful_files = len([f for f in st.session_state.processed_files if f['status'] == 'success'])
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">{successful_files}</div>
                <div class="metric-label">Successful</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            total_text_length = sum(f['text_length'] for f in st.session_state.processed_files)
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">{total_text_length:,}</div>
                <div class="metric-label">Total Characters</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.subheader("📋 File Details")
        df = pd.DataFrame(st.session_state.processed_files)
        st.dataframe(df, use_container_width=True)
        
        if successful_files > 0:
            st.subheader("📈 Text Length Distribution")
            successful_files_data = [f for f in st.session_state.processed_files if f['status'] == 'success']
            
            if successful_files_data:
                chart_data = pd.DataFrame({
                    'File': [f['name'] for f in successful_files_data],
                    'Text Length': [f['text_length'] for f in successful_files_data]
                })
                st.bar_chart(chart_data.set_index('File'))

# RUN THE APPLICATION
if __name__ == "__main__":
    main()
