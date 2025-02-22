import streamlit as st
import pandas as pd
import requests

from google import genai
from google.genai import types
import httpx

import json

def extract_data_from_text(response_text):

    try:
        json_str = response_text.split("```json")[1].split("```")[0].strip()
        data = json.loads(json_str)
    except Exception as e:
        st.error(f"Error extracting data: {str(e)}")
        data = {}

    return data