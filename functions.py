import whisper
import os
import random
import openai
import yt_dlp
from pytube import YouTube
import pandas as pd
import plotly_express as px
import nltk
import plotly.graph_objects as go
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import streamlit as st
import en_core_web_lg
import validators
import re
import itertools
import numpy as np
from bs4 import BeautifulSoup   
import base64, time
from annotated_text import annotated_text
import pickle, math
import wikipedia
from pyvis.network import Network
import torch
from pydub import AudioSegment
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings,HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import StdOutCallbackHandler
from langchain.chains import ConversationalRetrievalChain, QAGenerationChain, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

from langchain.prompts import PromptTemplate

nltk.download('punkt')


from nltk import sent_tokenize

OPEN_AI_KEY = os.environ.get('OPEN_AI_KEY')
time_str = time.strftime("%d%m%Y-%H%M%S")
HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; 
margin-bottom: 2.5rem">{}</div> """

memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')


#Stuff Chain Type Prompt template

@st.cache_data
def load_prompt():

    system_template="""Use only the following pieces of earnings context to answer the users question accurately.
    Do not use any information not provided in the earnings context and remember you are a to speak like a finance expert.
    If you don't know the answer, just say 'There is no relevant answer in the given earnings call transcript', 
    don't try to make up an answer.
    
    ALWAYS return a "SOURCES" part in your answer.
    The "SOURCES" part should be a reference to the source of the document from which you got your answer.
    
    Remember, do not reference any information not given in the context.
    If the answer is not available in the given context just say 'There is no relevant answer in the given earnings call transcript'
    
    Follow the below format when answering:
    
    Question: {question}
    SOURCES: [xyz]
      
    Begin!
    ----------------
    {context}"""
    
    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
    prompt = ChatPromptTemplate.from_messages(messages)
        
    return prompt

###################### Functions #######################################################################################

# @st.cache_data
# def get_yt_audio(url):
#     temp_audio_file = os.path.join('output', 'audio')

#     ydl_opts = {
#         'format': 'bestaudio/best',
#         'postprocessors': [{
#             'key': 'FFmpegExtractAudio',
#             'preferredcodec': 'mp3',
#             'preferredquality': '192',
#         }],
#         'outtmpl': temp_audio_file,
#         'quiet': True,
#     }

#     with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        
#         info = ydl.extract_info(url, download=False)
#         title = info.get('title', None)
#         ydl.download([url])

#     #with open(temp_audio_file+'.mp3', 'rb') as file:
#     audio_file = os.path.join('output', 'audio.mp3')

#     return audio_file, title

#load all required models and cache
@st.cache_resource
def load_models():

    '''Load and cache all the models to be used'''
    q_model = ORTModelForSequenceClassification.from_pretrained("nickmuchi/quantized-optimum-finbert-tone")
    ner_model = AutoModelForTokenClassification.from_pretrained("xlm-roberta-large-finetuned-conll03-english")
    kg_model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")
    kg_tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
    q_tokenizer = AutoTokenizer.from_pretrained("nickmuchi/quantized-optimum-finbert-tone")
    ner_tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large-finetuned-conll03-english")
    emb_tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-xl')
    sent_pipe = pipeline("text-classification",model=q_model, tokenizer=q_tokenizer)
    sum_pipe = pipeline("summarization",model="philschmid/flan-t5-base-samsum",clean_up_tokenization_spaces=True)
    ner_pipe = pipeline("ner", model=ner_model, tokenizer=ner_tokenizer, grouped_entities=True)
    cross_encoder = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1') #cross-encoder/ms-marco-MiniLM-L-12-v2
    sbert = SentenceTransformer('all-MiniLM-L6-v2')
    
    return sent_pipe, sum_pipe, ner_pipe, cross_encoder, kg_model, kg_tokenizer, emb_tokenizer, sbert

@st.cache_resource
def get_spacy():
    nlp = en_core_web_lg.load()
    return nlp
    
nlp = get_spacy()    

sent_pipe, sum_pipe, ner_pipe, cross_encoder, kg_model, kg_tokenizer, emb_tokenizer, sbert  = load_models()

@st.cache_data
def get_yt_audio(url):

    '''Get YT video from given URL link'''
    yt = YouTube(url)

    title = yt.title

    # Get the first available audio stream and download it
    audio_stream =  yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first().download()

    return audio_stream, title

@st.cache_data
def load_whisper_api(audio):

    '''Transcribe YT audio to text using Open AI API'''
    file = open(audio, "rb")
    transcript = openai.Audio.translate("whisper-1", file)

    return transcript

@st.cache_data
def load_asr_model(model_name):

    '''Load the open source  whisper model in cases where the API is not working'''
    model = whisper.load_model(model_name)

    return model

@st.cache_data
def inference(link, upload, _asr_model):
    '''Convert Youtube video or Audio upload to text'''
    
    try:
        
        if validators.url(link):

            st.info("`Downloading YT audio...`")
            
            audio_file, title = get_yt_audio(link)

            if 'audio' not in st.session_state:
                st.session_state['audio'] = audio_file
    
            #Get size of audio file
            audio_size = round(os.path.getsize(audio_file)/(1024*1024),1)
    
            #Check if file is > 24mb, if not then use Whisper API
            if audio_size <= 25:

                st.info("`Transcribing YT audio...`")
                
                #Use whisper API
                results = load_whisper_api(audio_file)['text']
    
            else:
    
                st.warning('File size larger than 24mb, applying chunking and transcription',icon="⚠️")
    
                song = AudioSegment.from_file(audio_file, format='mp4')
    
                # PyDub handles time in milliseconds
                twenty_minutes = 20 * 60 * 1000
                
                chunks = song[::twenty_minutes]
                
                transcriptions = []
                
                for i, chunk in enumerate(chunks):
                    chunk.export(f'output/chunk_{i}.mp4', format='mp4')
                    transcriptions.append(load_whisper_api(f'output/chunk_{i}.mp4')['text'])
    
                results = ','.join(transcriptions)

            st.info("`YT Video transcription process complete...`")

            return results, title

        elif _upload:

            #Get size of audio file
            audio_size = round(os.path.getsize(_upload)/(1024*1024),1)
    
            #Check if file is > 24mb, if not then use Whisper API
            if audio_size <= 25:

                st.info("`Transcribing uploaded audio...`")
                
                #Use whisper API
                results = load_whisper_api(_upload)['text']
    
            else:
    
                st.write('File size larger than 24mb, applying chunking and transcription')
    
                song = AudioSegment.from_file(_upload)
    
                # PyDub handles time in milliseconds
                twenty_minutes = 20 * 60 * 1000
                
                chunks = song[::twenty_minutes]
                
                transcriptions = []

                st.info("`Transcribing uploaded audio...`")
                
                for i, chunk in enumerate(chunks):
                    chunk.export(f'output/chunk_{i}.mp3', format='mp3')
                    transcriptions.append(load_whisper_api('output/chunk_{i}.mp3')['text'])
    
                results = ','.join(transcriptions)

            st.info("`Uploaded audio transcription process complete...`")

            return results, "Transcribed Earnings Audio"
                
    except Exception as e:

        st.error(f'''Whisper API Error: {e}, 
                    Using Whisper module from GitHub, might take longer than expected''',icon="🚨")
        
        results = _asr_model.transcribe(st.session_state['audio'], task='transcribe', language='en')
      
        return results['text'], title

@st.cache_data
def clean_text(text):
    '''Clean all text after inference'''

    text = text.encode("ascii", "ignore").decode()  # unicode
    text = re.sub(r"https*\S+", " ", text)  # url
    text = re.sub(r"@\S+", " ", text)  # mentions
    text = re.sub(r"#\S+", " ", text)  # hastags
    text = re.sub(r"\s{2,}", " ", text)  # over spaces
    
    return text

@st.cache_data
def chunk_long_text(text,threshold,window_size=3,stride=2):
    '''Preprocess text and chunk for sentiment analysis'''
    
    #Convert cleaned text into sentences
    sentences = sent_tokenize(text)
    out = []

    #Limit the length of each sentence to a threshold
    for chunk in sentences:
        if len(chunk.split()) < threshold:
            out.append(chunk)
        else:
            words = chunk.split()
            num = int(len(words)/threshold)
            for i in range(0,num*threshold+1,threshold):
                out.append(' '.join(words[i:threshold+i]))
    
    passages = []
    
    #Combine sentences into a window of size window_size
    for paragraph in [out]:
        for start_idx in range(0, len(paragraph), stride):
            end_idx = min(start_idx+window_size, len(paragraph))
            passages.append(" ".join(paragraph[start_idx:end_idx]))
            
    return passages  

@st.cache_data
def sentiment_pipe(earnings_text):
    '''Determine the sentiment of the text'''
    
    earnings_sentences = chunk_long_text(earnings_text,150,1,1)
    earnings_sentiment = sent_pipe(earnings_sentences)
    
    return earnings_sentiment, earnings_sentences 

@st.cache_data
def chunk_and_preprocess_text(text, model_name= 'philschmid/flan-t5-base-samsum'):

    '''Chunk and preprocess text for summarization'''
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    sentences = sent_tokenize(text)
    
    # initialize
    length = 0
    chunk = ""
    chunks = []
    count = -1
    
    for sentence in sentences:
        count += 1
        combined_length = len(tokenizer.tokenize(sentence)) + length # add the no. of sentence tokens to the length counter
    
        if combined_length  <= tokenizer.max_len_single_sentence: # if it doesn't exceed
            chunk += sentence + " " # add the sentence to the chunk
            length = combined_length # update the length counter
    
            # if it is the last sentence
            if count == len(sentences) - 1:
                chunks.append(chunk) # save the chunk
      
        else: 
            chunks.append(chunk) # save the chunk
            # reset 
            length = 0 
            chunk = ""
        
            # take care of the overflow sentence
            chunk += sentence + " "
            length = len(tokenizer.tokenize(sentence))

    return chunks

@st.cache_data
def summarize_text(text_to_summarize,max_len,min_len):
    '''Summarize text with HF model'''
    
    summarized_text = sum_pipe(text_to_summarize,
                               max_length=max_len,
                               min_length=min_len,
                               do_sample=False, 
                               early_stopping=True,
                              num_beams=4)
    summarized_text = ' '.join([summ['summary_text'] for summ in summarized_text])
     
    return summarized_text 

@st.cache_data 	
def get_all_entities_per_sentence(text):
    doc = nlp(''.join(text))

    sentences = list(doc.sents)

    entities_all_sentences = []
    for sentence in sentences:
        entities_this_sentence = []

        # SPACY ENTITIES
        for entity in sentence.ents:
            entities_this_sentence.append(str(entity))

        # XLM ENTITIES
        entities_xlm = [entity["word"] for entity in ner_pipe(str(sentence))]
        for entity in entities_xlm:
            entities_this_sentence.append(str(entity))

        entities_all_sentences.append(entities_this_sentence)

    return entities_all_sentences

@st.cache_data 
def get_all_entities(text):
    all_entities_per_sentence = get_all_entities_per_sentence(text)
    return list(itertools.chain.from_iterable(all_entities_per_sentence))

@st.cache_data    
def get_and_compare_entities(article_content,summary_output):
    
    all_entities_per_sentence = get_all_entities_per_sentence(article_content)
    entities_article = list(itertools.chain.from_iterable(all_entities_per_sentence))
   
    all_entities_per_sentence = get_all_entities_per_sentence(summary_output)
    entities_summary = list(itertools.chain.from_iterable(all_entities_per_sentence))
   
    matched_entities = []
    unmatched_entities = []
    for entity in entities_summary:
        if any(entity.lower() in substring_entity.lower() for substring_entity in entities_article):
            matched_entities.append(entity)
        elif any(
                np.inner(sbert.encode(entity, show_progress_bar=False),
                         sbert.encode(art_entity, show_progress_bar=False)) > 0.9 for
                art_entity in entities_article):
            matched_entities.append(entity)
        else:
            unmatched_entities.append(entity)

    matched_entities = list(dict.fromkeys(matched_entities))
    unmatched_entities = list(dict.fromkeys(unmatched_entities))

    matched_entities_to_remove = []
    unmatched_entities_to_remove = []

    for entity in matched_entities:
        for substring_entity in matched_entities:
            if entity != substring_entity and entity.lower() in substring_entity.lower():
                matched_entities_to_remove.append(entity)

    for entity in unmatched_entities:
        for substring_entity in unmatched_entities:
            if entity != substring_entity and entity.lower() in substring_entity.lower():
                unmatched_entities_to_remove.append(entity)

    matched_entities_to_remove = list(dict.fromkeys(matched_entities_to_remove))
    unmatched_entities_to_remove = list(dict.fromkeys(unmatched_entities_to_remove))

    for entity in matched_entities_to_remove:
        matched_entities.remove(entity)
    for entity in unmatched_entities_to_remove:
        unmatched_entities.remove(entity)

    return matched_entities, unmatched_entities

@st.cache_data 
def highlight_entities(article_content,summary_output):
   
    markdown_start_red = "<mark class=\"entity\" style=\"background: rgb(238, 135, 135);\">"
    markdown_start_green = "<mark class=\"entity\" style=\"background: rgb(121, 236, 121);\">"
    markdown_end = "</mark>"

    matched_entities, unmatched_entities = get_and_compare_entities(article_content,summary_output)
    
    print(summary_output)

    for entity in matched_entities:
        summary_output = re.sub(f'({entity})(?![^rgb\(]*\))',markdown_start_green + entity + markdown_end,summary_output)

    for entity in unmatched_entities:
        summary_output = re.sub(f'({entity})(?![^rgb\(]*\))',markdown_start_red + entity + markdown_end,summary_output)
    
    print("")
    print(summary_output)
    
    print("")
    print(summary_output)
    
    soup = BeautifulSoup(summary_output, features="html.parser")

    return HTML_WRAPPER.format(soup)

def summary_downloader(raw_text):
    '''Download the summary generated'''
    
    b64 = base64.b64encode(raw_text.encode()).decode()
    new_filename = "new_text_file_{}_.txt".format(time_str)
    st.markdown("#### Download Summary as a File ###")
    href = f'<a href="data:file/txt;base64,{b64}" download="{new_filename}">Click to Download!!</a>'
    st.markdown(href,unsafe_allow_html=True)

@st.cache_data
def generate_eval(raw_text, N, chunk):

    # Generate N questions from context of chunk chars
    # IN: text, N questions, chunk size to draw question from in the doc
    # OUT: eval set as JSON list

    # raw_text = ','.join(raw_text)
    
    update = st.empty()
    ques_update = st.empty()
    update.info("`Generating sample questions ...`")
    n = len(raw_text)
    starting_indices = [random.randint(0, n-chunk) for _ in range(N)]
    sub_sequences = [raw_text[i:i+chunk] for i in starting_indices]
    chain = QAGenerationChain.from_llm(ChatOpenAI(temperature=0))
    eval_set = []
    for i, b in enumerate(sub_sequences):
        qa = chain.run(b)
        eval_set.append(qa)
        ques_update.info(f"Creating Question: {i+1}")
        
    eval_set_full = list(itertools.chain.from_iterable(eval_set))
    
    update.empty()
    ques_update.empty()

    return eval_set_full

@st.cache_resource
def gen_embeddings(embedding_model):

    '''Generate embeddings for given model'''
    
    if 'hkunlp' in embedding_model:
        
        embeddings = HuggingFaceInstructEmbeddings(model_name=embedding_model,
                                           query_instruction='Represent the Financial question for retrieving supporting paragraphs: ',
                                           embed_instruction='Represent the Financial paragraph for retrieval: ')

    else:
        
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    return embeddings
    
@st.cache_data
def process_corpus(corpus, title, embedding_model, chunk_size=1000, overlap=50):

    '''Process text for Semantic Search'''
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=overlap)

    texts = text_splitter.split_text(corpus)

    embeddings = gen_embeddings(embedding_model)

    vectorstore = FAISS.from_texts(texts, embeddings, metadatas=[{"source": i} for i in range(len(texts))])

    return vectorstore
    
def embed_text(query,_docsearch):
    
    '''Embed text and generate semantic search scores'''
    
    # llm = OpenAI(temperature=0)
    chat_llm = ChatOpenAI(streaming=True,
                          model_name = 'gpt-4',
                          callbacks=[StdOutCallbackHandler()], 
                          verbose=True, 
                          temperature=0
                         )

    # chain = RetrievalQA.from_chain_type(llm=chat_llm, chain_type="stuff", 
    #                              retriever=_docsearch.as_retriever(), 
    #                              return_source_documents=True)
    
    question_generator = LLMChain(llm=chat_llm, prompt=CONDENSE_QUESTION_PROMPT) 
    doc_chain = load_qa_chain(llm=chat_llm,chain_type="stuff",prompt=load_prompt()) 
    chain = ConversationalRetrievalChain(retriever=_docsearch.as_retriever(search_kwags={"k": 3}), 
                                     question_generator=question_generator, 
                                     combine_docs_chain=doc_chain, 
                                     memory=memory, 
                                     return_source_documents=True, 
                                     get_chat_history=lambda h :h)
    
    answer = chain({"question": query})

    return answer

@st.cache_data
def gen_sentiment(text):
    '''Generate sentiment of given text'''
    return sent_pipe(text)[0]['label']

@st.cache_data 
def gen_annotated_text(df):
    '''Generate annotated text'''
    
    tag_list=[]
    for row in df.itertuples():
        label = row[2]
        text = row[1]
        if label == 'Positive':
            tag_list.append((text,label,'#8fce00'))
        elif label == 'Negative':
            tag_list.append((text,label,'#f44336'))
        else:
            tag_list.append((text,label,'#000000'))
        
    return tag_list
    
    
def display_df_as_table(model,top_k,score='score'):
    '''Display the df with text and scores as a table'''
    
    df = pd.DataFrame([(hit[score],passages[hit['corpus_id']]) for hit in model[0:top_k]],columns=['Score','Text'])
    df['Score'] = round(df['Score'],2)
    
    return df   

      
def make_spans(text,results):
    results_list = []
    for i in range(len(results)):
        results_list.append(results[i]['label'])
    facts_spans = []
    facts_spans = list(zip(sent_tokenizer(text),results_list))
    return facts_spans

##Fiscal Sentiment by Sentence
def fin_ext(text):
    results = remote_clx(sent_tokenizer(text))
    return make_spans(text,results)

## Knowledge Graphs code

@st.cache_data 
def extract_relations_from_model_output(text):
    relations = []
    relation, subject, relation, object_ = '', '', '', ''
    text = text.strip()
    current = 'x'
    text_replaced = text.replace("<s>", "").replace("<pad>", "").replace("</s>", "")
    for token in text_replaced.split():
        if token == "<triplet>":
            current = 't'
            if relation != '':
                relations.append({
                    'head': subject.strip(),
                    'type': relation.strip(),
                    'tail': object_.strip()
                })
                relation = ''
            subject = ''
        elif token == "<subj>":
            current = 's'
            if relation != '':
                relations.append({
                    'head': subject.strip(),
                    'type': relation.strip(),
                    'tail': object_.strip()
                })
            object_ = ''
        elif token == "<obj>":
            current = 'o'
            relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
    if subject != '' and relation != '' and object_ != '':
        relations.append({
            'head': subject.strip(),
            'type': relation.strip(),
            'tail': object_.strip()
        })
    return relations

def from_text_to_kb(text, model, tokenizer, article_url, span_length=128, article_title=None,
                    article_publish_date=None, verbose=False):
    # tokenize whole text
    inputs = tokenizer([text], return_tensors="pt")

    # compute span boundaries
    num_tokens = len(inputs["input_ids"][0])
    if verbose:
        print(f"Input has {num_tokens} tokens")
    num_spans = math.ceil(num_tokens / span_length)
    if verbose:
        print(f"Input has {num_spans} spans")
    overlap = math.ceil((num_spans * span_length - num_tokens) / 
                        max(num_spans - 1, 1))
    spans_boundaries = []
    start = 0
    for i in range(num_spans):
        spans_boundaries.append([start + span_length * i,
                                 start + span_length * (i + 1)])
        start -= overlap
    if verbose:
        print(f"Span boundaries are {spans_boundaries}")

    # transform input with spans
    tensor_ids = [inputs["input_ids"][0][boundary[0]:boundary[1]]
                  for boundary in spans_boundaries]
    tensor_masks = [inputs["attention_mask"][0][boundary[0]:boundary[1]]
                    for boundary in spans_boundaries]
    inputs = {
        "input_ids": torch.stack(tensor_ids),
        "attention_mask": torch.stack(tensor_masks)
    }

    # generate relations
    num_return_sequences = 3
    gen_kwargs = {
        "max_length": 256,
        "length_penalty": 0,
        "num_beams": 3,
        "num_return_sequences": num_return_sequences
    }
    generated_tokens = model.generate(
        **inputs,
        **gen_kwargs,
    )

    # decode relations
    decoded_preds = tokenizer.batch_decode(generated_tokens,
                                           skip_special_tokens=False)

    # create kb
    kb = KB()
    i = 0
    for sentence_pred in decoded_preds:
        current_span_index = i // num_return_sequences
        relations = extract_relations_from_model_output(sentence_pred)
        for relation in relations:
            relation["meta"] = {
                article_url: {
                    "spans": [spans_boundaries[current_span_index]]
                }
            }
            kb.add_relation(relation, article_title, article_publish_date)
        i += 1

    return kb

def get_article(url):
    article = Article(url)
    article.download()
    article.parse()
    return article

def from_url_to_kb(url, model, tokenizer):
    article = get_article(url)
    config = {
        "article_title": article.title,
        "article_publish_date": article.publish_date
    }
    kb = from_text_to_kb(article.text, model, tokenizer, article.url, **config)
    return kb

def get_news_links(query, lang="en", region="US", pages=1):
    googlenews = GoogleNews(lang=lang, region=region)
    googlenews.search(query)
    all_urls = []
    for page in range(pages):
        googlenews.get_page(page)
        all_urls += googlenews.get_links()
    return list(set(all_urls))

def from_urls_to_kb(urls, model, tokenizer, verbose=False):
    kb = KB()
    if verbose:
        print(f"{len(urls)} links to visit")
    for url in urls:
        if verbose:
            print(f"Visiting {url}...")
        try:
            kb_url = from_url_to_kb(url, model, tokenizer)
            kb.merge_with_kb(kb_url)
        except ArticleException:
            if verbose:
                print(f"  Couldn't download article at url {url}")
    return kb

def save_network_html(kb, filename="network.html"):
    # create network
    net = Network(directed=True, width="700px", height="700px")

    # nodes
    color_entity = "#00FF00"
    for e in kb.entities:
        net.add_node(e, shape="circle", color=color_entity)

    # edges
    for r in kb.relations:
        net.add_edge(r["head"], r["tail"],
                    title=r["type"], label=r["type"])

    # save network
    net.repulsion(
        node_distance=200,
        central_gravity=0.2,
        spring_length=200,
        spring_strength=0.05,
        damping=0.09
    )
    net.set_edge_smooth('dynamic')
    net.show(filename)

def save_kb(kb, filename):
    with open(filename, "wb") as f:
        pickle.dump(kb, f)

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'KB':
            return KB
        return super().find_class(module, name)

def load_kb(filename):
    res = None
    with open(filename, "rb") as f:
        res = CustomUnpickler(f).load()
    return res

class KB():
    def __init__(self):
        self.entities = {} # { entity_title: {...} }
        self.relations = [] # [ head: entity_title, type: ..., tail: entity_title,
          # meta: { article_url: { spans: [...] } } ]
        self.sources = {} # { article_url: {...} }

    def merge_with_kb(self, kb2):
        for r in kb2.relations:
            article_url = list(r["meta"].keys())[0]
            source_data = kb2.sources[article_url]
            self.add_relation(r, source_data["article_title"],
                              source_data["article_publish_date"])

    def are_relations_equal(self, r1, r2):
        return all(r1[attr] == r2[attr] for attr in ["head", "type", "tail"])

    def exists_relation(self, r1):
        return any(self.are_relations_equal(r1, r2) for r2 in self.relations)

    def merge_relations(self, r2):
        r1 = [r for r in self.relations
              if self.are_relations_equal(r2, r)][0]

        # if different article
        article_url = list(r2["meta"].keys())[0]
        if article_url not in r1["meta"]:
            r1["meta"][article_url] = r2["meta"][article_url]

        # if existing article
        else:
            spans_to_add = [span for span in r2["meta"][article_url]["spans"]
                            if span not in r1["meta"][article_url]["spans"]]
            r1["meta"][article_url]["spans"] += spans_to_add

    def get_wikipedia_data(self, candidate_entity):
        try:
            page = wikipedia.page(candidate_entity, auto_suggest=False)
            entity_data = {
                "title": page.title,
                "url": page.url,
                "summary": page.summary
            }
            return entity_data
        except:
            return None

    def add_entity(self, e):
        self.entities[e["title"]] = {k:v for k,v in e.items() if k != "title"}

    def add_relation(self, r, article_title, article_publish_date):
        # check on wikipedia
        candidate_entities = [r["head"], r["tail"]]
        entities = [self.get_wikipedia_data(ent) for ent in candidate_entities]

        # if one entity does not exist, stop
        if any(ent is None for ent in entities):
            return

        # manage new entities
        for e in entities:
            self.add_entity(e)

        # rename relation entities with their wikipedia titles
        r["head"] = entities[0]["title"]
        r["tail"] = entities[1]["title"]

        # add source if not in kb
        article_url = list(r["meta"].keys())[0]
        if article_url not in self.sources:
            self.sources[article_url] = {
                "article_title": article_title,
                "article_publish_date": article_publish_date
            }

        # manage new relation
        if not self.exists_relation(r):
            self.relations.append(r)
        else:
            self.merge_relations(r)

    def get_textual_representation(self):
        res = ""
        res += "### Entities\n"
        for e in self.entities.items():
            # shorten summary
            e_temp = (e[0], {k:(v[:100] + "..." if k == "summary" else v) for k,v in e[1].items()})
            res += f"- {e_temp}\n"
        res += "\n"
        res += "### Relations\n"
        for r in self.relations:
            res += f"- {r}\n"
        res += "\n"
        res += "### Sources\n"
        for s in self.sources.items():
            res += f"- {s}\n"
        return res
            
def save_network_html(kb, filename="network.html"):
    # create network
    net = Network(directed=True, width="700px", height="700px", bgcolor="#eeeeee")

    # nodes
    color_entity = "#00FF00"
    for e in kb.entities:
        net.add_node(e, shape="circle", color=color_entity)

    # edges
    for r in kb.relations:
        net.add_edge(r["head"], r["tail"],
                    title=r["type"], label=r["type"])
        
    # save network
    net.repulsion(
        node_distance=200,
        central_gravity=0.2,
        spring_length=200,
        spring_strength=0.05,
        damping=0.09
    )
    net.set_edge_smooth('dynamic')
    net.show(filename)
