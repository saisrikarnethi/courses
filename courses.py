import os
import faiss
import pandas as pd
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.document_loaders import TextLoader
import gradio as gr

# Step 1: Load Data
def load_data(file_path):
    """Load course data from a CSV file."""
    return pd.read_csv(file_path)

# Step 2: Generate Embeddings
def create_embeddings(data, column="description"):
    """Generate embeddings for the course descriptions."""
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    texts = data[column].tolist()
    metadata = data[["title", "url"]].to_dict(orient="records")
    vector_store = FAISS.from_texts(texts, embeddings, metadatas=metadata)
    vector_store.save_local("vector_store")
    return vector_store

# Step 3: Load Vector Store
def load_vector_store():
    """Load the vector store from disk."""
    return FAISS.load_local("vector_store", HuggingFaceEmbeddings())

# Step 4: Create Search Functionality
def search_courses(query, vector_store):
    """Search for relevant courses based on the query."""
    retriever = vector_store.as_retriever()
    llm = OpenAI(temperature=0)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain.run(query)

# Step 5: Gradio Interface
def create_interface():
    """Create a Gradio interface for the smart search tool."""
    def query_interface(user_query):
        vector_store = load_vector_store()
        return search_courses(user_query, vector_store)

    interface = gr.Interface(
        fn=query_interface,
        inputs=gr.Textbox(label="Search for Courses"),
        outputs=gr.Textbox(label="Search Results"),
        title="Smart Course Search",
        description="Search Analytics Vidhya's Free Courses with AI-powered Smart Search"
    )
    return interface

if __name__ == "__main__":
    # Ensure the vector store is prepared
    if not os.path.exists("vector_store"):  # Check if embeddings already exist
        data = load_data("courses.csv")  # Replace with your data path
        create_embeddings(data)

    # Launch the interface
    app = create_interface()
    app.launch()
