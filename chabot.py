import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS  # Vector embeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate  # Helps create prompts
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set. Check your .env file!") # Raise an error if the API key is not set

# Function to extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf_doc in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf_doc)
            for page in pdf_reader.pages:  # Iterate through each page
                text += page.extract_text() or ""  # Extract text from the PDF, handle None
        except Exception as e:
            st.error(f"Error reading PDF: {pdf_doc.name}. Error: {e}")
            return ""  # Return empty string if there's an issue with a file
    return text

# Function to split the text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Reduced chunk size
        chunk_overlap=200,  # Reduced overlap
        length_function=len,
        is_separator_regex=False,
    )  # Split text
    chunks = text_splitter.split_text(text)
    return chunks

# Function to store text chunks as embeddings
def get_vector_store(text_chunks):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)  # Create vector store
        vector_store.save_local("faiss_index")  # Save the vector store
        return vector_store  # Return the vector store
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None

# Function to create a conversational AI chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. 
    If the answer is not available in the provided context, just say, "answer is not available in the context". 
    Do not provide a wrong answer. \n\n
    Context:\n {context}\n
    Question:\n {question}\n

    Answer:
    """

    #initialize the model
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

    #create prompt using the template
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    #load the QA chain
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain

# Function to process user input and retrieve answers
def user_input(user_question, vector_store):  # Added vector_store as argument
    if vector_store is None:
        st.error("Vector store is not initialized.")
        return

    try:
        docs = vector_store.similarity_search(user_question)  # Search for similar documents
        chain = get_conversational_chain()  # Create conversational AI chain
        response = chain(
            {"input_documents": docs, "question": user_question}, return_only_outputs=True
        )
        print(response)
        st.write("Reply:", response["output_text"])  # Print the response
    except Exception as e:
        st.error(f"Error processing user input: {e}")

# Streamlit UI
def main():
    st.set_page_config(page_title="Chat with Multiple PDF Files", layout="wide") # Set page title and layout
    st.header("Chat with Multiple PDFs using Gemini!")

    # Initialize the vector store in session_state if it doesn't exist
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question and st.session_state.vector_store:
        user_input(user_question, st.session_state.vector_store)  # Pass vector_store from session_state

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button",
            accept_multiple_files=True,
        )
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                if not raw_text:  # Check if raw_text is empty
                    st.error("No text extracted from the uploaded PDFs.")
                    return
                text_chunks = get_text_chunks(raw_text)
                vector_store = get_vector_store(text_chunks)  # Get vector store
                if vector_store:
                    st.session_state.vector_store = vector_store  # Store in session_state
                    st.success("Done")
                else:
                    st.error("Failed to process the documents.")

if __name__ == "__main__":
    main()







