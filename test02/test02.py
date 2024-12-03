import os
from dotenv import load_dotenv
load_dotenv("/home/vboxuser/.env")
HF_API_KEY = os.getenv("HF_API_KEY")

#!pip install sentencepiece

# 모델과 토크나이저 로드

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = 'Bllossom/llama-3.2-Korean-Bllossom-3B'
#model_id = 'LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct'

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)


def chatbot_response(user_input):
#    inputs = tokenizer(user_input, return_tensors="pt")
#    outputs = model.generate(inputs, max_length=150, num_return_sequences=1)
# response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    messages = [
        {"role": "user", "content": f"{user_input}"}
        ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.convert_tokens_to_ids("<|end_of_text|>"),
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
        input_ids,
        max_new_tokens=1024,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

#print(tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True))

def main():
    print("챗봇에 오신 것을 환영합니다! '종료'라고 입력하면 대화가 종료됩니다.")
    while True:
        user_input = input("당신: ")
        if user_input.lower() == "종료":
            print("챗봇을 종료합니다. 안녕히 가세요!")
            break
        response = chatbot_response(user_input)
        print("챗봇:", response)

if __name__ == "__main__":
    main()
