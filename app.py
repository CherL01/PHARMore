import streamlit as st
import pandas as pd
import requests

from google import genai
from google.genai import types
import httpx

import json
import os

GENAI_KEY = os.getenv("GENAI_KEY")

def extract_data_from_text(response_text):

    try:
        json_str = response_text.split("```json")[1].split("```")[0].strip()
        data = json.loads(json_str)
    except Exception as e:
        st.error(f"Error extracting data: {str(e)}")
        data = {}

    return data

client = genai.Client(api_key=GENAI_KEY)

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
            "Literature": "PMID or DOI of the original paper or report.",
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
        "instruction": "I will provide the medical report or paper. Please provide the extracted information. If any information is not present, please state 'Not stated'. Do not return anything except for the required fields.",
        }
        """

st.title("PDF or URL Metadata Extractor")

upload_type = st.radio('Select input type:', ('Upload a PDF File', 'Enter a URL'))


if upload_type == 'Upload a PDF File':
    uploaded_file = st.file_uploader("Choose a PDF file", type='pdf')

    if uploaded_file is not None:
        # file_name = uploaded_file.name
        
        response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[
            prompt,
            types.Part.from_bytes(
                data=uploaded_file.getvalue(),
                mime_type='application/pdf',
            )])

        response = extract_data_from_text(response.text)
        df = pd.DataFrame([response])
        st.write("Extracted PDF Metadata:", df)

elif upload_type == 'Enter a URL':
    url = st.text_input('Enter the URL:')
    
    if url:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                # title = response.text.split("<title>")[1].split("</title>")[0] if "<title>" in response.text else "No Title"
                
                doc_data = httpx.get(url).content

                response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[
                    prompt,
                    types.Part.from_bytes(
                        data=doc_data,
                        mime_type='application/pdf',
                    )])

                response = extract_data_from_text(response.text)
                df = pd.DataFrame([response])
                st.write("Extracted URL Metadata:", response)
            else:
                st.error(f"Failed to fetch the URL, status code: {response.status_code}")
        except Exception as e:
            st.error(f"Error fetching URL: {str(e)}")
