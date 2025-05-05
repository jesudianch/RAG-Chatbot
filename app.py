# Import required libraries
import fitz  # PyMuPDF for PDF processing
import tempfile  # For temporary file handling
import gradio as gr  # For creating the web interface

# Import LangChain components for RAG implementation
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline

# Import Hugging Face pipeline for text generation
from transformers import pipeline

def load_pdf_chunks(file_path, chunk_size=500, chunk_overlap=50):
    """
    Load and process a PDF file into text chunks.
    
    Args:
        file_path (str): Path to the PDF file
        chunk_size (int): Size of each text chunk
        chunk_overlap (int): Number of characters to overlap between chunks
    
    Returns:
        list: List of Document objects containing text chunks
    """
    # Open and read the PDF file
    doc = fitz.open(file_path)
    # Extract text from all pages
    text = "\n".join([page.get_text() for page in doc])
    # Initialize text splitter with specified chunk size and overlap
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    # Split text into chunks
    chunks = splitter.split_text(text)
    # Create Document objects for each chunk with metadata
    return [Document(page_content=chunk, metadata={"source": file_path}) for chunk in chunks if chunk.strip()]

def setup_rag(documents):
    """
    Set up the RAG (Retrieval-Augmented Generation) pipeline.
    
    Args:
        documents (list): List of Document objects containing text chunks
    
    Returns:
        RetrievalQA: Configured RAG chain for question answering
    """
    # Initialize embeddings using Hugging Face model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # Create vector store from documents
    vectorstore = FAISS.from_documents(documents, embeddings)
    # Configure retriever with MMR (Maximal Marginal Relevance) search
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 8, "lambda_mult": 0.5}
    )
    # Initialize text generation pipeline
    gen_pipeline = pipeline("text2text-generation", model="google/flan-t5-base", max_length=128)
    # Create LLM from pipeline
    llm = HuggingFacePipeline(pipeline=gen_pipeline)
    # Create and return RAG chain
    chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
    return chain

# Global variable to store the RAG chain
qa_chain = None

def upload_pdf(file):
    """
    Handle PDF upload and indexing.
    
    Args:
        file: Uploaded PDF file object
    
    Returns:
        str: Status message
    """
    global qa_chain
    # Get the path of the uploaded file
    pdf_path = file.name
    # Process PDF into chunks
    docs = load_pdf_chunks(pdf_path)
    # Set up RAG pipeline with the documents
    qa_chain = setup_rag(docs)
    return "PDF uploaded and indexed!"

def query_rag(question):
    """
    Process a question through the RAG pipeline.
    
    Args:
        question (str): User's question
    
    Returns:
        str: Generated answer or error message
    """
    if qa_chain is None:
        return "Upload a PDF first!"
    # Get answer from RAG chain
    result = qa_chain({"query": question})
    return result["result"]

# Create Gradio interface
with gr.Blocks() as demo:
    # Add title
    gr.Markdown("## 🧠 RAG App with MMR + PDF Upload (Hugging Face Demo)")
    
    # PDF upload section
    with gr.Row():
        file = gr.File(label="Upload a PDF", file_types=[".pdf"])
        upload_btn = gr.Button("Upload and Index")
    status = gr.Textbox(label="Status")
    upload_btn.click(upload_pdf, inputs=file, outputs=status)

    # Question-answer section
    with gr.Row():
        question = gr.Textbox(label="Enter your question")
        answer = gr.Textbox(label="Answer")
        answer_btn = gr.Button("Answer")
    answer_btn.click(query_rag, inputs=question, outputs=answer)

if __name__ == "__main__":
    # Launch the Gradio interface
    demo.launch()