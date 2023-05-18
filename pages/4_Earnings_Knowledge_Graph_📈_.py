import streamlit as st
from pyvis.network import Network
from functions import *
import streamlit.components.v1 as components
import pickle, math

st.set_page_config(page_title="Earnings Knowledge Graph", page_icon="📈")
st.sidebar.header("Knowledge Graph")
st.markdown("## Earnings Knowledge Graph")

filename = "earnings_network.html"

if "earnings_passages" in st.session_state:

    with st.spinner(text='Loading Babelscape/rebel-large which can take a few minutes to generate the graph..'):
    
        st.session_state.kb_text = from_text_to_kb(st.session_state['earnings_passages'], kg_model, kg_tokenizer, "", verbose=True)
        save_network_html(st.session_state.kb_text, filename=filename)
        st.session_state.kb_chart = filename

    with st.container():
        st.subheader("Generated Knowledge Graph")
        st.markdown("*You can interact with the graph and zoom.*")
        html_source_code = open(st.session_state.kb_chart, 'r', encoding='utf-8').read()
        components.html(html_source_code, width=700, height=700)
        st.markdown(st.session_state.kb_text)

else:

    st.write('No earnings text detected, please regenerate from Home page..')
