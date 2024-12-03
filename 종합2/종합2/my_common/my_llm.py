# "test_rag_all_llm_general.ipynb" 에서 복사

# LLM template 


import anthropic
from openai import OpenAI
#from langchain_community.llms.ollama import Ollama
from langchain_ollama import OllamaLLM
#from langchain_community.chat_models import ChatOllama
#from langchain_anthropic import ChatAnthropic
#from langchain_openai import ChatOpenAI
#
from abc import ABC, abstractmethod
from langchain.prompts import ChatPromptTemplate
# 실행시간을 측정하는 모듈
import time

# RAG형식을 처리위한, 프롬프트 템플릿

PROMPT_TEMPLATE = """
Basing only on the following context:

{context}

---

Answer the following question: {question}
Avoid to start the answer saying that you are basing on the provided context and go straight with the response.
"""

PROMPT_TEMPLATE_NAIVE = """
Answer the following question: {question}
"""

PROMPT_TEMPLATE_EMPTY = """
{question}
"""

class LLM(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def invoke(self, prompt: str) -> str:
        pass

    def generate_response(self, context: str = None, question: str = None, is_template=True, qa_template = None) -> str:
        prompt = None
        if context is not None:
            print("-------1--------")
            prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
            prompt = prompt_template.format(context=context, question=question)
        elif qa_template is not None:
            print("-------2--------",qa_template)
            promt = qa_template
        elif is_template:
            print("-------3--------")
            prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE_NAIVE)
            prompt = prompt_template.format(question=question)
        else:
            print("-------4--------", question)
            #prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE_EMPTY)
            #prompt = prompt_template.format(question=question)
            prompt = question
        #
        #print(prompt)
        
        response_text, elapsed_time = self.invoke(prompt)
        return response_text, elapsed_time

class OllamaModel(LLM):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.model = OllamaLLM(model=model_name)
        self.model_name = model_name

    def invoke(self, prompt: str) -> str:
        start_time = time.time()
        response = self.model.invoke(prompt)
        elapsed_time = time.time() - start_time
        
        return response, elapsed_time

    def __del__(self):
        self.model = OllamaLLM(model=self.model_name, keep_alive=0)
        #print("deleted..")
        
#To unload the model and free up memory use:
#curl http://localhost:11434/api/generate -d '{"model": "llama3.2", "keep_alive": 0}'
        
class GPTModel(LLM):
    def __init__(self, model_name: str, api_key: str):
        super().__init__(model_name)
        self.client = OpenAI(api_key=api_key)

    def invoke(self, prompt: str) -> str:
        messages = [
            #{"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        start_time = time.time()
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0.7,
        )
        elapsed_time = time.time() - start_time

        return response.choices[0].message.content.strip(), elapsed_time
    
class AnthropicModel(LLM):
    def __init__(self, model_name: str, api_key: str):
        super().__init__(model_name)
        self.client = anthropic.Anthropic(api_key=api_key)

    def invoke(self, prompt: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]

        start_time = time.time()
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=1000,
            temperature=0.7,
            messages=messages
        )
        # Extract the plain text from the response content
        text_blocks = response.content
        plain_text = "\n".join(block.text for block in text_blocks if block.type == 'text')
        elapsed_time = time.time() - start_time
        
        return plain_text, elapsed_time


# 1) 표준 템플릿
# "test_llm_general.ipynb"과 "test_llm_rag_general.ipynb"에서 복사 및 수정한 코드

#############################################################
# 0) 선언 부분

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class LocalModel(LLM):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # 결과값을 보여주는 template
        if self.tokenizer.chat_template is None:
            self.tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}{{ ' ' }}{% endif %}{{ message['content'] }}{% if not loop.last %}{{ '  ' }}{% endif %}{% endfor %}{{ eos_token }}"
        #if tokenizer.pad_token is None:
        #    tokenizer.pad_token = tokenizer.eos_token
        #if tokenizer.pad_token_id is None:
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        # 에러해결
        # The attention mask is not set and cannot be inferred from input because pad token is same as eos token. 
        # As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
        
        self.client = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            #trust_remote_code=True,  # exaone only
        )
        # 'unsloth/Llama-3.2-1B-Instruct 사용시에는 다음을 막아야 함.
        if model_name == 'meta-llama/Llama-3.2-1B':
            self.client.generation_config.pad_token_id = self.client.generation_config.eos_token_id
            self.client.generation_config.pad_token_id = self.tokenizer.pad_token_id   # 설정하지 않으면, 무한 CPU 실행
        
    def invoke(self, prompt: str) -> str:
        #############################################################
        # 1) prompt과정
        messages = [
            #{"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        
        #############################################################
        # 2) tokenizer과정
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.client.device)
               
        #############################################################
        # 3) LLM과정
        start_time = time.time()
        outputs = self.client.generate(
            input_ids,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.9,
            top_p=0.9,
            pad_token_id = self.tokenizer.eos_token_id,  # llama 3.2, bllossom
        )
        elapsed_time = time.time() - start_time
        
        #############################################################
        # 4) decoder과정
        answer = self.tokenizer.decode(outputs[0])
    
        # 특수 토근을 제거하고, 출력
        response = self.tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
    
        return response, elapsed_time

# LLM facotry

class LLMFactory:
    @staticmethod
    def create_llm(model_type: str, model_name: str, api_key: str = None) -> LLM:
        if model_type == 'local':
            return LocalModel(model_name)
        elif model_type == 'ollama':
            return OllamaModel(model_name)
        elif model_type == 'gpt':
            return GPTModel(model_name, api_key)
        elif model_type == 'claude':
            return AnthropicModel(model_name, api_key)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")