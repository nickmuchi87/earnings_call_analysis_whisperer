import streamlit as st
import pandas as pd
import plotly_express as px
import plotly.graph_objects as go
from functions import *
import validators
import textwrap

#st.set_page_config(page_title="Earnings Sentiment Analysis", page_icon="📈")
st.sidebar.header("Sentiment Analysis")
st.markdown("## Earnings Sentiment Analysis with FinBert-Tone")

#load whisper model
asr_model = load_asr_model(st.session_state.sbox)

if "url" not in st.session_state:
    st.session_state.url = ''

if "title" not in st.session_state:
    st.session_state.title = ''   

try:

    if st.session_state['url'] is not None or st.session_state['upload'] is not None:
        
        results, title = inference(st.session_state.url,st.session_state.upload,asr_model)

        print(f'results, page1: {results}')
    
        st.subheader(title)
        
        earnings_passages = clean_text(results)
        
        st.session_state['earnings_passages'] = earnings_passages

        st.session_state['title'] = title
            
        earnings_sentiment, earnings_sentences = sentiment_pipe(earnings_passages)
        
        with st.expander("See Transcribed Earnings Text"):
            st.write(f"Number of Sentences: {len(earnings_sentences)}")
            
            st.write(st.session_state['earnings_passages'])
        
        
        ## Save to a dataframe for ease of visualization
        sen_df = pd.DataFrame(earnings_sentiment)
        sen_df['text'] = earnings_sentences
        grouped = pd.DataFrame(sen_df['label'].value_counts()).reset_index()
        grouped.columns = ['sentiment','count']
        
        st.session_state['sen_df'] = sen_df
        
        # Display number of positive, negative and neutral sentiments
        fig = px.bar(grouped, x='sentiment', y='count', color='sentiment', color_discrete_map={"Negative":"firebrick","Neutral":\
                                                                                               "navajowhite","Positive":"darkgreen"},\
                                                                                               title='Earnings Sentiment')
        
        fig.update_layout(
        	showlegend=False,
            autosize=True,
            margin=dict(
                l=25,
                r=25,
                b=25,
                t=50,
                pad=2
            )
        )
        
        
        st.plotly_chart(fig)
        
        ## Display sentiment score
        pos_perc = grouped[grouped['sentiment']=='Positive']['count'].iloc[0]*100/sen_df.shape[0]
        neg_perc = grouped[grouped['sentiment']=='Negative']['count'].iloc[0]*100/sen_df.shape[0]
        neu_perc = grouped[grouped['sentiment']=='Neutral']['count'].iloc[0]*100/sen_df.shape[0]
        
        sentiment_score = neu_perc+pos_perc-neg_perc
        
        fig_1 = go.Figure()
        
        fig_1.add_trace(go.Indicator(
            mode = "delta",
            value = sentiment_score,
            domain = {'row': 1, 'column': 1}))
        
        fig_1.update_layout(
        	template = {'data' : {'indicator': [{
                'title': {'text': "Sentiment Score"},
                'mode' : "number+delta+gauge",
                'delta' : {'reference': 50}}]
                                 }},
            autosize=False,
            width=250,
            height=250,
            margin=dict(
                l=5,
                r=5,
                b=5,
                pad=2
            )
        )
        
        with st.sidebar:
        
            st.plotly_chart(fig_1)

        hd = sen_df.text.apply(lambda txt: '<br>'.join(textwrap.wrap(txt, width=70)))
        ## Display negative sentence locations
        fig = px.scatter(sen_df, y='label', color='label', size='score', hover_data=[hd], color_discrete_map={"Negative":"firebrick","Neutral":"navajowhite","Positive":"darkgreen"}, title='Sentiment Score Distribution')
        
        
        fig.update_layout(
        	showlegend=False,
            autosize=True,
            width=800,
            height=500,
            margin=dict(
                b=5,
                t=50,
                pad=4
            )
        )
        
        st.plotly_chart(fig)
        
    else:
    
        st.write("No YouTube URL or file upload detected")
        
except (AttributeError, TypeError):

    st.write("No YouTube URL or file upload detected")
