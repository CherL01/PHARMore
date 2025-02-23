import streamlit as st
import pandas as pd
import requests
import httpx
import json
import os

from google import genai
from google.genai import types

# from drug_discovery import train_autoencoder, get_similar_drugs_autoencoder

GENAI_KEY = os.getenv("GENAI_KEY")
client = genai.Client(api_key=GENAI_KEY)

st.markdown("""
    <style>
        .stTabs [role="tablist"] {
            margin-bottom: 20px;
        }
        .stTabs [role="tab"] {
            color: #edf2fb;
        }
    </style>
""", unsafe_allow_html=True)

def extract_data_from_text(response_text):
    """Extract JSON data wrapped in markdown formatting from the response."""
    if "```json" not in response_text:
        st.error("The response did not contain the expected JSON formatting.")
        return {}
    try:
        json_str = response_text.split("```json")[1].split("```")[0].strip()
        data = json.loads(json_str)
    except Exception as e:
        st.error(f"Error extracting data: {str(e)}")
        data = {}
    return data

def generate_extraction(content_bytes, mime_type):
    """Generate content extraction using the GenAI API."""
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[
            prompt,
            types.Part.from_bytes(data=content_bytes, mime_type=mime_type)
        ]
    )
    return extract_data_from_text(response.text)

@st.cache_data(show_spinner=False)
def fetch_url_content(url):
    """Fetch and return the content from a URL if it appears to be a PDF."""
    try:
        response = requests.get(url)
        if response.status_code == 200:
            # title = response.text.split("<title>")[1].split("</title>")[0] if "<title>" in response.text else "No Title"
            
            doc = httpx.get(url).content
        return doc
    except Exception as e:
        st.error(f"Error fetching URL: {str(e)}")
        return None

prompt = """
{
"task": "You are a medical and genetic expert. I am providing you with a medical paper or report regarding a genetic variant. Your task is to extract the following key information from the text.",
"fields": {
    "Variant": "The identifier of the genetic variant (e.g., rsID like rs113993960).",
    "Genes": "The gene(s) associated with the variant (e.g., CFTR).",
    "Drugs": "The drug(s) or treatment(s) associated with the variant or condition.",
    "Association": "The relationship between the genetic variant and the associated condition or phenotype.",
    "Significance": "The reported significance of the association (e.g., not stated, significant, etc.).",
    "P-Value": "The p-value associated with the statistical analysis of the variant's significance.",
    "Number of Cases": "The number of cases or individuals with the condition.",
    "Number of Controls": "The number of controls or individuals without the condition.",
    "Biogeographical Groups": "Information on the biogeographical groups or populations analyzed.",
    "Phenotype Categories": "The phenotype categories or traits related to the variant.",
    "Pediatric": "Any details regarding pediatric (children) cases or studies mentioned.",
    "More Details": "Any additional details, such as mechanisms, biological processes, etc.",
    "Literature": "PMID or DOI of the original paper or report."
},
"example_report": {
    "Variant": "rs113993960",
    "Genes": "CFTR",
    "Drugs": "ivacaftor / lumacaftor",
    "Association": "Genotype del/del is associated with decreased severity of Exocrine Pancreatic Insufficiency when treated with ivacaftor / lumacaftor in children with Cystic Fibrosis.",
    "Significance": "not stated",
    "P-Value": "1",
    "Number of Cases": "0",
    "Number of Controls": "Unknown",
    "Biogeographical Groups": "Efficacy",
    "Phenotype Categories": "PMIID:34511391"
},
"instruction": "I will provide the medical report or paper. Please provide the extracted information. If any information is not present, please state 'Not stated'. Do not return anything except for the required fields."
}
"""

st.markdown("""
    <div style="text-align: center; margin-bottom: 20px;">
        <p style="color: #48cae4; font-size: 3rem; margin-bottom: 10px;">PHARMore 💊</p>
        <p style="color: #00b4d8; font-size: 2rem; margin-bottom: 5px;">Drug Discovery and Medical Research Assistant</p>
        <p style="font-size: 1.0rem; color: #0096c7;">Extract detailed metadata from medical reports and papers. Answer drug and diease related queries.</p>
    </div>
""", unsafe_allow_html=True)

tabs = st.tabs(["PDF File", "URL", "Chatbot"]) #, "Drug Discovery"])

# ---------------- PDF Tab ----------------
with tabs[0]:
    st.subheader("Upload a PDF File")
    uploaded_file = st.file_uploader("Choose a PDF file", type='pdf')
    if uploaded_file is not None:
        with st.spinner("Extracting metadata from PDF..."):
            file_bytes = uploaded_file.getvalue()
            extracted_data = generate_extraction(file_bytes, mime_type='application/pdf')
        if extracted_data:
            df = pd.DataFrame([extracted_data])
            st.markdown('<div style="background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">', unsafe_allow_html=True)
            st.write("Extracted PDF Metadata:", df)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.error("No data was extracted from the PDF.")

# ---------------- URL Tab ----------------
with tabs[1]:
    st.subheader("Enter a URL")
    url = st.text_input('Enter the URL:')
    if url:
        with st.spinner("Fetching and processing URL content..."):
            doc_data = fetch_url_content(url)
            if doc_data:
                extracted_data = generate_extraction(doc_data, mime_type='application/pdf')
                if extracted_data:
                    df = pd.DataFrame([extracted_data])
                    st.markdown('<div style="background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">', unsafe_allow_html=True)
                    st.write("Extracted URL Metadata:", df)
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.error("No data was extracted from the URL content.")
            else:
                st.error("Failed to fetch content from the URL.")

# ---------------- Chatbot Tab ----------------
with tabs[2]:
    st.subheader("Medical Chatbot")
    st.write("Ask any drug or disease medical question below and get a response.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    with st.form("chat_form", clear_on_submit=True):
        user_question = st.text_input("Your Question:")
        submitted = st.form_submit_button("Send")
        if submitted and user_question:
            st.session_state.chat_history.append({"role": "user", "content": user_question})

            chat_prompt = (
                f"You are a knowledgeable pharmaceutical expert. You will give medical advice that will not have legal consequences. Answer the following question concisely:\n\n"
                f"Question: {user_question}\nAnswer:"
            )

            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[chat_prompt]
            )
            bot_answer = response.text.strip()

            st.session_state.chat_history.append({"role": "bot", "content": bot_answer})

    if st.session_state.chat_history:
        for chat in st.session_state.chat_history:
            if chat["role"] == "user":
                st.markdown(f"**You:** {chat['content']}")
            else:
                st.markdown(f"**Bot:** {chat['content']}")

# # ---------------- Drug Discovery Tab ----------------
# with tabs[3]:
#     st.subheader("Drug Discovery")
#     st.write("Enter a PubChem id (ie. 54675785) to find a similar drug using our deep learning model. Drugs are returned in the format [DRUG_ID: SIMILARITY_SCORE].")

#     with st.form("drug_form", clear_on_submit=True):
#         drug_input = st.text_input("Enter a PubChem id:")
#         submitted_drug = st.form_submit_button("Find Similar Drug")
#         if submitted_drug and drug_input:
#             sim_df = train_autoencoder()
#             similar_drug = get_similar_drugs_autoencoder(drug_input, sim_df)
#             st.write(f"Drug query: {drug_input} | Similar drugs:")
#             st.table(similar_drug)

# import streamlit as st
# import pandas as pd
# import requests

# from google import genai
# from google.genai import types
# import httpx

# import json
# import os

# GENAI_KEY = os.getenv("GENAI_KEY")

# def extract_data_from_text(response_text):

#     try:
#         json_str = response_text.split("```json")[1].split("```")[0].strip()
#         data = json.loads(json_str)
#     except Exception as e:
#         st.error(f"Error extracting data: {str(e)}")
#         data = {}

#     return data

# client = genai.Client(api_key=GENAI_KEY)

# prompt = """
#         {
#         "task": "You are a medical and genetic expert. I am providing you with a medical paper or report regarding a genetic variant. Your task is to extract the following key information from the text.",
#         "fields": {
#             "Variant": "The identifier of the genetic variant (e.g., rsID like rs113993960).",
#             "Genes": "The gene(s) associated with the variant (e.g., CFTR).",
#             "Drugs": "The drug(s) or treatment(s) associated with the variant or condition.",
#             "Association": "The relationship between the genetic variant and the associated condition or phenotype.",
#             "Significance": "The reported significance of the association (e.g., not stated, significant, etc.).",
#             "P-Value": "The p-value associated with the statistical analysis of the variant's significance.",
#             "Number of Cases": "The number of cases or individuals with the condition.",
#             "Number of Controls": "The number of controls or individuals without the condition.",
#             "Biogeographical Groups": "Information on the biogeographical groups or populations analyzed.",
#             "Phenotype Categories": "The phenotype categories or traits related to the variant.",
#             "Pediatric": "Any details regarding pediatric (children) cases or studies mentioned.",
#             "More Details": "Any additional details, such as mechanisms, biological processes, etc.",
#             "Literature": "PMID or DOI of the original paper or report.",
#         },
#         "example_report": {
#             "Variant": "rs113993960",
#             "Genes": "CFTR",
#             "Drugs": "ivacaftor / lumacaftor",
#             "Association": "Genotype del/del is associated with decreased severity of Exocrine Pancreatic Insufficiency when treated with ivacaftor / lumacaftor in children with Cystic Fibrosis.",
#             "Significance": "not stated",
#             "P-Value": "1",
#             "Number of Cases": "0",
#             "Number of Controls": "Unknown",
#             "Biogeographical Groups": "Efficacy",
#             "Phenotype Categories": "PMIID:34511391"
#         },
#         "instruction": "I will provide the medical report or paper. Please provide the extracted information. If any information is not present, please state 'Not stated'. Do not return anything except for the required fields.",
#         }
#         """

# st.markdown("""
#     <div style="text-align: center; margin-bottom: 20px;">
#         <h1 style="color: #48cae4; font-size: 3rem; margin-bottom: 10px;">PHARMore 💊</h1>
#         <h2 style="color: #00b4d8; font-size: 2rem; margin-bottom: 5px;">Drug Discovery and Medical Research Assistant</h2>
#         <p style="font-size: 1.2rem; color: #0096c7;">Extract detailed metadata from medical reports and papers.</p>
#     </div>
# """, unsafe_allow_html=True)

# st.title("PHARMore 💊")
# st.subheader("Drug Discovery and Medical Research Assistant")
# st.write("Extract detailed metadata from medical reports and papers.")

# upload_type = st.radio('Select input type:', ('Upload a PDF File', 'Enter a URL'))


# if upload_type == 'Upload a PDF File':
#     uploaded_file = st.file_uploader("Choose a PDF file", type='pdf')

#     if uploaded_file is not None:
#         # file_name = uploaded_file.name
        
#         response = client.models.generate_content(
#         model="gemini-2.0-flash",
#         contents=[
#             prompt,
#             types.Part.from_bytes(
#                 data=uploaded_file.getvalue(),
#                 mime_type='application/pdf',
#             )])

#         response = extract_data_from_text(response.text)
#         df = pd.DataFrame([response])
#         st.write("Extracted PDF Metadata:", df)

# elif upload_type == 'Enter a URL':
#     url = st.text_input('Enter the URL:')
    
#     if url:
#         try:
#             response = requests.get(url)
#             if response.status_code == 200:
#                 # title = response.text.split("<title>")[1].split("</title>")[0] if "<title>" in response.text else "No Title"
                
#                 doc_data = httpx.get(url).content

#                 response = client.models.generate_content(
#                 model="gemini-2.0-flash",
#                 contents=[
#                     prompt,
#                     types.Part.from_bytes(
#                         data=doc_data,
#                         mime_type='application/pdf',
#                     )])

#                 response = extract_data_from_text(response.text)
#                 df = pd.DataFrame([response])
#                 st.write("Extracted URL Metadata:", df)
#             else:
#                 st.error(f"Failed to fetch the URL, status code: {response.status_code}")
#         except Exception as e:
#             st.error(f"Error fetching URL: {str(e)}")