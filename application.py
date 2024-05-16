import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
import os
from pptx import Presentation
from docx import Document 
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import pandas as pd
load_dotenv()
os.getenv("OPENAI_API_KEY")

import streamlit as st
import zipfile
import io
# Other imports remain the same

def extract_zip_files(zip_file):
    extracted_files = []
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall('temp_extracted_files')
        for file_name in zip_ref.namelist():
            extracted_files.append('temp_extracted_files/' + file_name)
    return extracted_files

def get_text_from_word(word_docs):
    text = ""
    for word in word_docs:
        doc = Document(word)
        for para in doc.paragraphs:
            text += para.text
    return text
    
def get_text_from_excel(excel_docs):
    text = ""
    for excel in excel_docs:
        df = pd.read_excel(excel)
        text += " ".join(df.stack().astype(str))
    return text
    
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_from_ppt(ppt_docs):
    text = ""
    for ppt in ppt_docs:
        try:
            prs = Presentation(ppt)
            for slide in prs.slides:
                for shape in slide.shapes:
                    if hasattr(shape, "text"):
                        text += shape.text
        except Exception as e:
            st.warning(f"Error processing {ppt.name}: {str(e)}")
    return text

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, chunk_overlap=1000)
    chunks = splitter.split_text(text)
    return chunks  # list of strings

def get_vectorstore(chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    vectorstore.save_local("faiss_index")
 

def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "upload a folder containing pdfs/ppts/word/excel docs and ask me a question"}]

def get_conversation_chain():
    prompt_template = """
    You are a professional in understanding content from pdfs and answering any questions related to it.You are also a helpful assistant who knows to greet people. Understand the content provided and answer the questions appropriately. Ensure that the sentences you provide are grammatically correct. Answer the question as detailed as possible from the provided context for all questions except greetings, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer. If it is greetings please answer appropriately\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatOpenAI()
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = OpenAIEmbeddings()  # type: ignore

    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True) 
    docs = new_db.similarity_search(user_question)

    chain = get_conversation_chain()

    response = chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True, )

    print(response)
    return response


def main():
    st.set_page_config(
        page_title="OpenAI PDF/PPT Chatbot",
        page_icon="book"
    )

    # Sidebar for uploading PDF files
    # Sidebar for uploading folder
    # Update the sidebar for zip file upload
    with st.sidebar:
        st.title("Menu:")
        uploaded_zip_file = st.file_uploader("Upload your ZIP File", type=['zip'])
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                if uploaded_zip_file is not None:
                    extracted_files = extract_zip_files(io.BytesIO(uploaded_zip_file.read()))
                    pdf_docs = [doc for doc in extracted_files if doc.lower().endswith('.pdf')]
                    ppt_docs = [doc for doc in extracted_files if doc.lower().endswith(('.ppt', '.pptx'))]
                    word_docs = [doc for doc in extracted_files if doc.lower().endswith(('.doc', '.docx'))]    
                    pdf_text = get_pdf_text(pdf_docs)
                    ppt_text = get_text_from_ppt(ppt_docs)
                    word_text = get_text_from_word(word_docs)
                    
                    combined_text = pdf_text + ppt_text + word_text
                    text_chunks = get_text_chunks(combined_text)
                    get_vectorstore(text_chunks)
                    st.success("Done")
    # Main content area for displaying chat messages
    st.title("Chat with PDF files")
    st.write("Welcome to the chat!")
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    # Chat input
    # Placeholder for chat messages

    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": "upload some pdfs and ask me a question"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

      # Display chat messages and bot response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = user_input(prompt)
                placeholder = st.empty()
                full_response = ''
                for item in response['output_text']:
                    full_response += item
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)
        if response is not None:
            message = {"role": "assistant", "content": full_response}
            st.session_state.messages.append(message)


if __name__ == "__main__":
    main()
