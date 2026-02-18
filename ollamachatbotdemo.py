#streamlit + langchain + LLMs + ollama(LLM-gemma2:2b model)
#import required libraries

import os
import streamlit as st 

#import pyton built in os module  

from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

#step 1 create prompt template 
#this define how ai should behave & how it receives user input 

prompt = ChatPromptTemplate.from_messages(
    [
        #system message defines ai behaviour 
        ("system","you are a helpful assistant, please respond clearly to the question asked"),
        #user message contains placeholder {question}
        ("user","Question : {question}")
    ]
)

# step 2 - streamlit app ui

#app title 
st.title("langchain demo with gemma model(ollama)")

#text input box for user question
input_txt = st.text_input("what question do you have in your mind?")

#step 3 - load ollama model 

#load local gemma model 
LLM = Ollama(model="gemma3:4b")

#condition -  convert output model to string
output_parser = StrOutputParser()

#create langchain pipeline (prompt --> model --> parser)
chain = prompt | LLM | output_parser

#step 4 - run the model when user input question
if input_txt : 
    response = chain.invoke({"question":input_txt})
    st.write(response)
    