import streamlit as st
from utils.pdf_loader import load_and_chunk_pdfs
from utils.embedder import embed_chunks, build_faiss_index, retrieve_chunks
from utils.qa_engine import answer_with_roberta
from sentence_transformers import SentenceTransformer
import faiss
import tempfile
import os

st.title("📘 MBA – SmartBot")
st.markdown("Upload a case study PDF and ask business questions!")

# Load embedding & QA models
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Global session state to store chunks & embeddings
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "index" not in st.session_state:
    st.session_state.index = None

# Upload PDFs
uploaded_files = st.file_uploader("📤 Upload case PDF(s)", type="pdf", accept_multiple_files=True)

if uploaded_files:
    with st.spinner("Processing documents..."):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_paths = []
            for file in uploaded_files:
                path = os.path.join(tmpdir, file.name)
                with open(path, "wb") as f:
                    f.write(file.read())
                file_paths.append(path)

            # Process PDFs
            chunks = load_and_chunk_pdfs(file_paths)
            st.session_state.chunks = chunks
            st.session_state.embeddings = embed_chunks(chunks, embed_model)
            st.session_state.index = build_faiss_index(st.session_state.embeddings)

    st.success("✅ Case studies embedded and ready!")

    # After file is uploaded, show the text input for questions
    query = st.text_input("💬 Ask a business question from your uploaded case:")

    if query and st.session_state.index is not None:
        # Get relevant chunks and generate an answer
        top_chunks = retrieve_chunks(query, embed_model, st.session_state.index, st.session_state.chunks)
        answer = answer_with_roberta(query, top_chunks)

        # Display answer
        st.markdown("### 🧠 Answer:")
        st.write(answer)

        # Display relevant context used for the answer
        with st.expander("🔍 Context used from case(s)"):
            for i, chunk in enumerate(top_chunks):
                st.markdown(f"**Chunk {i+1}:** {chunk[:500]}...")

else:
    st.markdown("Please upload a PDF case study to begin.")