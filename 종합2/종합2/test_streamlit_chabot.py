# 다음 소스를 보고 수정한버전이다.

# https://medium.com/@eric_vaillancourt/mastering-langchain-rag-integrating-chat-history-part-2-4c80eae11b43
# https://github.com/Tolulade-A/opensource-llm-chatbot-blenderbot/blob/main/Chatbot_with_Open_Source_LLM_%26_Hugging_Face.ipynb

# 한글은 안되고, 영어는 잘된다.
# 2024.12.01

# 실행후, 소스를 수정하면 즉시 반영된것으로 동작된다.


import streamlit as st
from streamlit_chat import message
import requests
import time

import json
from dotenv import load_dotenv
load_dotenv("/home/vboxuser/.env")

import os

HF_TOKEN = os.getenv('HF_TOKEN')
print(HF_TOKEN)
#

API_URL = "https://api-inference.huggingface.co/models/facebook/blenderbot-400M-distill"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}


###
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

model_name = "facebook/blenderbot-400M-distill"
model = BlenderbotForConditionalGeneration.from_pretrained(model_name)
tokenizer = BlenderbotTokenizer.from_pretrained(model_name)

###
st.header("🤖BlenderBot (Demo)")

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    print(f"inputs={payload['inputs']}")
    input_ids = tokenizer(payload["inputs"]["text"], return_tensors="pt")
    outputs = model.generate(**input_ids)
    #print(tokenizer.batch_decode(outputs))
    response = tokenizer.batch_decode(outputs[0])
    print(f"response={response}")
    return response.json()


def query_new(payload):
    #print(payload)
    prompt = payload

    input_ids = tokenizer([prompt], return_tensors="pt",)
    #
    start_time = time.time()
    outputs = model.generate(
        **input_ids,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.9,
        top_p=0.9,
        pad_token_id = tokenizer.eos_token_id,  # llama 3.2, bllossom
    )
    elapsed_time = time.time() - start_time
    #
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    #print(response)

    return response[0]


with st.form('form', clear_on_submit=True):
    user_input = st.text_input('You: ', '', key='input')
    submitted = st.form_submit_button('Send')

if submitted and user_input:
    output = query_new(user_input)
    #print(output)
    st.session_state.past.append(user_input)
    #st.session_state.generated.append(output["generated_text"])
    st.session_state.generated.append(output)

if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
        message(st.session_state["generated"][i], key=str(i))

