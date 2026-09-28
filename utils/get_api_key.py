
import os
import streamlit as st
from groq import Groq
from mistralai.client import Mistral
from openai import OpenAI

@st.cache_resource
def get_groq_client() -> Groq:
    api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
    print("Bien recuperer la cle de Groq")
    if not api_key:
        st.error("Clé API Groq manquante. Configurez GROQ_API_KEY.")
        st.stop()
    return Groq(api_key=api_key)

@st.cache_resource
def get_mistral_client() -> Mistral:
    api_key = st.secrets.get("MISTRAL_API_KEY") or os.environ.get("MISTRAL_API_KEY")
    print("Bien recuperer la cle de Mistral")
    if not api_key:
        st.error("Clé API Mistral manquante. Configurez MISTRAL_API_KEY.")
        st.stop()
    
    return Mistral(api_key=api_key)


@st.cache_resource
def get_mistral_client_via_OpenAI() -> OpenAI:
    """" Obetenir la clé de Mistral via OpenAI  """
    api_key = st.secrets.get("MISTRAL_API_KEY") or os.environ.get("MISTRAL_API_KEY")
    print("Bien recuperer la cle de Mistral")
    if not api_key:
        st.error("Clé API Mistral manquante. Configurez MISTRAL_API_KEY.")
        st.stop()
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.mistral.ai/v1", 
    )
    return client

@st.cache_resource
def get_groq_client_via_OpenAI() -> OpenAI:
    """Client Groq exposé via l'interface compatible OpenAI."""
    api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
    if not api_key:
        st.error("Clé API Groq manquante. Configurez GROQ_API_KEY.")
        st.stop()
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )
    return client







