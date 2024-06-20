import streamlit as st
import os
import pypdf
from PIL import Image  
from dotenv import load_dotenv
from Models import get_HF_embeddings, cosine, get_doc2vec_embeddings

load_dotenv()  ## load all our environment variables

# Import the Generative AI model if needed
import google.generativeai as genai
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_gemini_response(input):
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(input)
    return response.text


def input_pdf_text(uploaded_files):
    
    reader = pypdf.PdfReader(uploaded_files)
    text = ""
    for page in range(len(reader.pages)):
        page = reader.pages[page]
        text += str(page.extract_text())
    return text

@st.cache_data
def get_key_info(text):
    """
    Extracts key information from the resume using Gemini
    """
    key_info_prompt = """
    This is a resume for a job applicant. Please analyze the text and provide the following information in a short and concise format. 
    do not elaborate the  Work Experience/internships details  . 
    resume:{text}
    
    I want the  response in paragraphs format  
    {{
    "Name":"",
    "Education": [],
    "Work Experience/internships": [],
    "Skills": [],
    }}
    """
    response = get_gemini_response(key_info_prompt.format(text='\n'.join(text)))
    return response

# Prompt Template
input_prompt = """
Hey Act Like a skilled or very experience ATS(Application Tracking System)
with a deep understanding of tech field,software engineering. Your task is to evaluate the resume based on the given job description.
You must consider the job market is very competitive. Assign the percentage Matching based 
on Job description and
the missing keywords with high accuracy
resume:{text}
description:{JD}


I want the below response in paragraphs format  
{{"JD Match":"%",
"MissingKeywords:[]",
}}
"""

def compare(resume_texts, JD_text, embedding_method='HuggingFace-BERT'):
    if embedding_method == 'Gemini':
        response = get_gemini_response(input_prompt.format(text='\n'.join(resume_texts), JD=JD_text))
        return response
    elif embedding_method == 'HuggingFace-BERT':
        JD_embeddings = get_HF_embeddings(JD_text)
        resume_embeddings = [get_HF_embeddings(resume_text) for resume_text in resume_texts]
         
        
    elif embedding_method == 'Doc2Vec':
        JD_embeddings, resume_embeddings = get_doc2vec_embeddings(JD_text, resume_texts)
       
    else:
        return "Invalid embedding method selected."

    cos_scores = cosine(resume_embeddings, JD_embeddings)
    return cos_scores[0]


## streamlit app
st.title("Smart Applicant Tracking System")



# Radio buttons for user choice
choice = st.radio("Choose Input Type", ('Folder', 'PDF'))

if choice == 'Folder':
    folder_path = st.text_input("Enter the path to the folder:")
    if folder_path:
        if os.path.isdir(folder_path):
            st.success(f"Folder path '{folder_path}' is valid.")
            for filename in os.listdir(folder_path):
              if filename.endswith(".pdf"):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, "rb") as pdf_file:
                    text = input_pdf_text(pdf_file)
                    key_info = get_key_info(text)
                    st.subheader(f"Key Information from {filename}:")
                    st.write(key_info)
        
        else:
            st.error(f"Folder path '{folder_path}' is not valid.")
else:
    pdf_file = st.file_uploader("Upload a PDF file", type="pdf")
    if pdf_file:
        st.success(f"PDF file '{pdf_file.name}' uploaded successfully.")

# Define uploaded_file outside the tab selection
uploaded_file = st.file_uploader(
    'Choose your resume.pdf file: ', type="pdf", help="Please upload the pdf"
)

# Tab selection
tab_selection = st.radio("Select Functionality", ["Extract key information", "Track Resume"])

if tab_selection == "Extract key information":
    if uploaded_file:
        st.subheader("Resume Uploaded Successfully!")
        text = input_pdf_text(uploaded_file)
        key_info = get_key_info(text)
        st.subheader("Key Information:")
        st.write(key_info)
    else:
        st.subheader("Please upload a resume ")

elif tab_selection == "Track Resume":
    if uploaded_file:
        JD = st.text_area("**Enter the job description:**")
        embedding_method = st.selectbox("Select Embedding Method", ['Gemini', 'HuggingFace-BERT', 'Doc2Vec'])

        submit = st.button("Submit")

        if submit:
            text = input_pdf_text(uploaded_file)
            response = compare([text], JD, embedding_method)
            st.subheader(response)
    else:
        st.subheader("Please upload a resume .")
