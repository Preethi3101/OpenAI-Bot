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
load_dotenv()
os.getenv("OPENAI_API_KEY")

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
    chain=get_conversation_chain()

    response = chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True, )

    return response

def main():
    st.set_page_config(
        page_title="OpenAI PDF/PPT Chatbot",
        page_icon="book"
    )

    # Sidebar for uploading folder
    with st.sidebar:
        st.title("Menu:")
        folder_uploaded = st.file_uploader("Upload a folder:", type="folder")
        if folder_uploaded:
            files = st.file_uploader("Upload your Excel Files and Click on the Submit & Process Button", accept_multiple_files=True)
            if st.button("Submit & Process"):
                with st.spinner("Processing..."):
                    pdf_docs = [file for file in files if file.name.lower().endswith('.pdf')]
                    ppt_docs = [file for file in files if file.name.lower().endswith(('.ppt', '.pptx'))]
                    word_docs = [file for file in files if file.name.lower().endswith(('.doc', '.docx'))]
                    excel_docs = [file for file in files if file.name.lower().endswith(('.xls', '.xlsx'))]
                    pdf_text = get_pdf_text(pdf_docs)
                    ppt_text = get_text_from_ppt(ppt_docs)
                    word_text = get_text_from_word(word_docs)
                    excel_text = get_text_from_excel(excel_docs)
                    combined_text = pdf_text + ppt_text + word_text + excel_text
                    text_chunks = get_text_chunks(combined_text)
                    get_vectorstore(text_chunks)
                    st.success("Folder uploaded and processed successfully")

    # Main content area for displaying chat messages
    st.title("OPENAI CHATBOT")
    st.write("Welcome to the chat!")
    if st.sidebar.button('Clear Chat History'):
        clear_chat_history()

    # Placeholder for chat messages
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Upload a folder containing PDFs, PPTs, Word docs, and Excel files, and then ask me a question."}
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.text_input("You:", key="user_input"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Display chat messages and bot response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = user_input(prompt)
                if response is not None:
                    st.write(response)

if __name__ == "__main__":
    main()
