# Earnings Call Analysis Whisperer
Transcription and Analysis of earnings calls from Youtube using the following:

This app assists finance analysts with transcribing and analysis Earnings Calls by carrying out the following tasks:

  - Transcribing earnings calls using Open AI's Whisper API, takes approx 3mins to transcribe a 1hr call less than 25mb in size.
  - Analysing the sentiment of transcribed text using the quantized version of [FinBert-Tone](https://huggingface.co/nickmuchi/quantized-optimum-finbert-tone).
  - Summarization of the call with [philschmid/flan-t5-base-samsum](https://huggingface.co/philschmid/flan-t5-base-samsum) model with entity extraction
  - Question Answering Search engine powered by Langchain, Open API (GPT-4) and [Sentence Transformers](https://huggingface.co/sentence-transformers/all-mpnet-base-v2).
  *Knowledge Graph generation using [Babelscape/rebel-large](https://huggingface.co/Babelscape/rebel-large) model.
  
