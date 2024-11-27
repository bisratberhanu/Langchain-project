import streamlit as st
import google.generativeai as geneai
from dotenv import load_dotenv
import logging
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA


load_dotenv()
geneai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize the Google Generative AI model
model = geneai.GenerativeModel("gemini-pro")

# Set up logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


@st.cache_data
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain(retriever):
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not
    provided in the context, just say, "answer is not available in the context". Do not provide the wrong answer.\n\n
    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chain = RetrievalQA.from_chain_type(
        llm=model,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        memory=memory,
    )
    return chain

def user_input(user_question):
    try:
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        # Load FAISS index and create retriever
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        retriever = new_db.as_retriever()

        # Create chain with retriever
        chain = get_conversational_chain(retriever)

        # Retrieve documents
        docs = retriever.invoke({"query": user_question})
        if not docs or "documents" not in docs:
            raise ValueError("No documents retrieved or invalid retriever output.")

        # Execute chain
        response = chain.invoke({"input_documents": docs["documents"], "query": user_question})

        # Update session state
        st.session_state["chat_history"].append((user_question, response["output_text"]))
        if len(st.session_state["chat_history"]) > 3:
            st.session_state["chat_history"] = st.session_state["chat_history"][-3:]

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        st.error("An error occurred while processing your request. Please try again later.")



def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    if "chat_history" in st.session_state:
        st.write("**Conversation History :**")
        for question, answer in st.session_state["chat_history"]:
            st.write(f"**👩‍💼 :** {question}")
            st.write(f"**🤖 :** {answer}")
            st.write("---")

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()