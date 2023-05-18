import whisper
import os
import pandas as pd
import plotly_express as px
import nltk
import plotly.graph_objects as go
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import streamlit as st
import en_core_web_lg

nltk.download('punkt')

from nltk import sent_tokenize

auth_token = os.environ.get("auth_token")

st.sidebar.header("Home")

asr_model_options = ['tiny.en','base.en','small.en']
    
asr_model_name = st.sidebar.selectbox("Whisper Model Options", options=asr_model_options, key='sbox')

st.markdown("## Earnings Call Analysis Whisperer")

twitter_link = """
[![](https://img.shields.io/twitter/follow/nickmuchi?label=@nickmuchi&style=social)](https://twitter.com/nickmuchi)
"""

st.markdown(twitter_link)

st.markdown(
    """
    This app assists finance analysts with transcribing and analysis Earnings Calls by carrying out the following tasks:
    - Transcribing earnings calls using Open AI's Whisper API, takes approx 3mins to transcribe a 1hr call less than 25mb in size.
    - Analysing the sentiment of transcribed text using the quantized version of [FinBert-Tone](https://huggingface.co/nickmuchi/quantized-optimum-finbert-tone).
    - Summarization of the call with [philschmid/flan-t5-base-samsum](https://huggingface.co/philschmid/flan-t5-base-samsum) model with entity extraction
    - Question Answering Search engine powered by Langchain and [Sentence Transformers](https://huggingface.co/sentence-transformers/all-mpnet-base-v2).
    - Knowledge Graph generation using [Babelscape/rebel-large](https://huggingface.co/Babelscape/rebel-large) model.
    
    **👇 Enter a YouTube Earnings Call URL below and navigate to the sidebar tabs** 
    
"""
)

if 'sbox' not in st.session_state:
    st.session_state.sbox = asr_model_name
    
if "url" not in st.session_state:
    st.session_state.url = ""
    
if "earnings_passages" not in st.session_state:
    st.session_state["earnings_passages"] = ''
    
if "sen_df" not in st.session_state:
    st.session_state['sen_df'] = ''
        
url_input = st.text_input(
        label="Enter YouTube URL, example below is McDonalds Earnings Call Q1 2023",
        value = "https://www.youtube.com/watch?v=4p6o5kkZYyA")

st.session_state['url'] = url_input
        
st.markdown(
    "<h3 style='text-align: center; color: red;'>OR</h3>",
    unsafe_allow_html=True
)

upload_wav = st.file_uploader("Upload a .wav/.mp3/.mp4 audio file ",key="upload",type=['.wav','.mp3','.mp4'])

st.markdown("![visitor badge](https://visitor-badge.glitch.me/badge?page_id=nickmuchi-earnings-call-whisperer)")
