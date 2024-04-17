from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai 
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import speech_recognition as sr
from dotenv import load_dotenv
#from wtforms import FileField
import os
from flask import Flask,request,render_template,jsonify,flash
import logging
from io import BytesIO

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

from docx import Document
import textract  # Install this library using 'pip install textract'

def get_text_from_pdf(pdf):
    """
    Extracts text from a PDF file.

    Args:
        pdf (str): Path to the PDF file.

    Returns:
        str: Extracted text.
    """
    try:
        # Read the PDF file
        text = textract.process(pdf)
        return text.decode("utf-8")
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

def get_text_from_docx(docx_file):
    """
    Extracts text from a DOCX file.

    Args:
        docx_file (str): Path to the DOCX file.

    Returns:
        str: Extracted text.
    """
    try:
        doc = Document(docx_file)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
    except Exception as e:
        print(f"Error extracting text from DOCX: {e}")
        return ""

def get_text_from_txt(txt_file):
    """
    Reads text from a TXT file.

    Args:
        txt_file (str): Path to the TXT file.

    Returns:
        str: Extracted text.
    """
    try:
        with open(txt_file, "r", encoding="utf-8") as file:
            text = file.read()
        return text
    except Exception as e:
        print(f"Error reading text from TXT: {e}")
        return ""
    
    
    
def get_text_from_file(file_path):
    _, file_extension = os.path.splitext(file_path.lower())
    if file_extension == ".pdf":
        return get_text_from_pdf(file_path)
    elif file_extension == ".docx":
        return get_text_from_docx(file_path)
    elif file_extension == ".txt":
        return get_text_from_txt(file_path)
    else:
        print(f"Unsupported file format: {file_extension}")
        return ""



def get_conversation():
    prompt_temp="""Answer the question as detailed as possible from the provided context. and handle the greetings like hii and response should be hii how can i help you If the answer is not in the provided 
    context, simply state, "The answer is not available Pleas Ask Questions Regarding to Decentrawood." Please avoid providing incorrect information.
    if a {{questions}} having different different variations with similar {{context}} or Meaning you should give the same answers.


Context:
{context}?

Question:
{question}

Answer:
    """

    model=ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.3)

    prompts=PromptTemplate(template=prompt_temp,input_variables=["context","question"])
    chain=load_qa_chain(model,chain_type="stuff",prompt=prompts)

    return chain

def get_pdf_text(pdf):
    # Read the PDF file from the BytesIO object
    pdf_reader =  PdfReader(pdf)
    text = ""
    for page in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page].extract_text()

    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store  # Return the vector store without saving it locally

def user_input(user_question,path):
    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # Add logging to check user_question
    logging.debug(f"User Question: {user_question}")

    # Load the FAISS model directly (without saving it to disk)
    text=get_text_from_file(path)
    text_chunks=get_text_chunks(text)
    new_db = get_vector_store(text_chunks)

    docs = new_db.similarity_search(user_question)

    chain = get_conversation()

    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    response_text = response["output_text"]

    return response_text


i=input()
result=user_input(i,r"C:\jayproject\texttoimage\Decentrawood Questions.txt")
print(result)


'''app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
@app.route('/user', methods=['POST'])
def handle_user_input():
    user_question = request.form.get('user_question')


    #Text input
    response_text=user_input(user_question)

    return jsonify({'response_text': response_text})'''







'''if __name__ == '__main__':
    app.run(debug=True)'''
