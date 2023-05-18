import streamlit as st
from functions import *

st.set_page_config(page_title="Earnings Summarization", page_icon="📖")
st.sidebar.header("Earnings Summarization")
st.markdown("## Earnings Summarization with Flan-T5-Base-SamSun")

max_len= st.slider("Maximum length of the summarized text",min_value=70,max_value=200,step=10,value=100)
min_len= st.slider("Minimum length of the summarized text",min_value=20,max_value=200,step=10)

st.markdown("####")     
        
st.subheader("Summarized Earnings Call with matched Entities")

if "earnings_passages" not in st.session_state:
    st.session_state["earnings_passages"] = ''

if st.session_state['earnings_passages']:
      
    with st.spinner("Summarizing and matching entities, this takes a few seconds..."):
        
        try:
            text_to_summarize = chunk_and_preprocess_text(st.session_state['earnings_passages'])
            print(text_to_summarize)
            summarized_text = summarize_text(text_to_summarize,max_len=max_len,min_len=min_len)
            
        
        except IndexError:
            try:
                
                text_to_summarize = chunk_and_preprocess_text(st.session_state['earnings_passages'])
                summarized_text = summarize_text(text_to_summarize,max_len=max_len,min_len=min_len)
                
    
            except IndexError:
                
                text_to_summarize = chunk_and_preprocess_text(st.session_state['earnings_passages'])
                summarized_text = summarize_text(text_to_summarize,max_len=max_len,min_len=min_len)
                        
        entity_match_html = highlight_entities(text_to_summarize,summarized_text)
        st.markdown("####")
        
        with st.expander(label='Summarized Earnings Call',expanded=True): 
            st.write(entity_match_html, unsafe_allow_html=True)
        
        st.markdown("####")     
        
        summary_downloader(summarized_text)
            
else:
      st.write("No text to summarize detected, please ensure you have entered the YouTube URL on the Sentiment Analysis page")
