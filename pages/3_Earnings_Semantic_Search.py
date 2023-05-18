import streamlit as st
from functions import *
from langchain.chains import QAGenerationChain
import itertools


st.set_page_config(page_title="Earnings Question/Answering", page_icon="🔎")

st.sidebar.header("Semantic Search")

st.markdown("Earnings Semantic Search with LangChain, OpenAI & SBert")

st.markdown(
    """
    <style>
    
    #MainMenu {visibility: hidden;
    # }
        footer {visibility: hidden;
        }
        .css-card {
            border-radius: 0px;
            padding: 30px 10px 10px 10px;
            background-color: black;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 10px;
            font-family: "IBM Plex Sans", sans-serif;
        }
        
        .card-tag {
            border-radius: 0px;
            padding: 1px 5px 1px 5px;
            margin-bottom: 10px;
            position: absolute;
            left: 0px;
            top: 0px;
            font-size: 0.6rem;
            font-family: "IBM Plex Sans", sans-serif;
            color: white;
            background-color: green;
            }
            
        .css-zt5igj {left:0;
        }
        
        span.css-10trblm {margin-left:0;
        }
        
        div.css-1kyxreq {margin-top: -40px;
        }
        
       
   
        
      
    </style>
    """,
    unsafe_allow_html=True,
)

bi_enc_dict = {'mpnet-base-v2':"all-mpnet-base-v2",
              'instructor-base': 'hkunlp/instructor-base'}

search_input = st.text_input(
        label='Enter Your Search Query',value= "What key challenges did the business face?", key='search')
        
sbert_model_name = st.sidebar.selectbox("Embedding Model", options=list(bi_enc_dict.keys()), key='sbox')

st.sidebar.markdown('Earnings QnA Generator')
        
chunk_size = 1000
overlap_size = 50

try:

    if search_input:
        
        if "sen_df" in st.session_state and "earnings_passages" in st.session_state:
        
            ## Save to a dataframe for ease of visualization
            sen_df = st.session_state['sen_df']

            title = st.session_state['title']

            earnings_text = st.session_state['earnings_passages']

            print(f'earnings_to_be_embedded:{earnings_text}')

            st.session_state.eval_set = generate_eval(
            earnings_text, 10, 3000)

            # Display the question-answer pairs in the sidebar with smaller text
            for i, qa_pair in enumerate(st.session_state.eval_set):
                st.sidebar.markdown(
                    f"""
                    <div class="css-card">
                    <span class="card-tag">Question {i + 1}</span>
                        <p style="font-size: 12px;">{qa_pair['question']}</p>
                        <p style="font-size: 12px;">{qa_pair['answer']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            embedding_model = bi_enc_dict[sbert_model_name]
                            
            with st.spinner(
                text=f"Loading {embedding_model} embedding model and Generating Response..."
            ):
                print('cheeky')
                print(earnings_text)
                docsearch = process_corpus(earnings_text,title, embedding_model)

                result = embed_text(search_input,docsearch)


            references = [doc.page_content for doc in result['source_documents']]

            answer = result['answer']

            sentiment_label = gen_sentiment(answer)
                
            ##### Sematic Search #####
            
            df = pd.DataFrame.from_dict({'Text':[answer],'Sentiment':[sentiment_label]})
              
            
            text_annotations = gen_annotated_text(df)[0]            
            
            with st.expander(label='Query Result', expanded=True):
                annotated_text(text_annotations)
                
            with st.expander(label='References from Corpus used to Generate Result'):
                for ref in references:
                    st.write(ref)
                
        else:
            
            st.write('Please ensure you have entered the YouTube URL or uploaded the Earnings Call file')
            
    else:
    
        st.write('Please ensure you have entered the YouTube URL or uploaded the Earnings Call file')  
        
except RuntimeError:
  
    st.write('Please ensure you have entered the YouTube URL or uploaded the Earnings Call file')    
