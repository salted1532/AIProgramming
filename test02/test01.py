import openai
import os
# OpenAI API 키 설정
#openai.api_key = 'YOUR_API_KEY'
from dotenv import load_dotenv

"""
cat ~/.env 
OPENAI_API_KEY="sk-proj-TX5N2TcTcIkIuW9B7H-05EeJyTyyqeEgA"
HF_API_KEY="hf_IXQTlBYReiFCYEihELiuxxxxEDNfSttu"
"""

load_dotenv("/home/vboxuser/.env")
openai.api_key = os.getenv("OPENAI_API_KEY")

def chat_with_bot(user_input):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": user_input}
        ]
    )
    return response['choices'][0]['message']['content']

if __name__ == "__main__":
    print("챗봇과 대화하세요! '종료'를 입력하면 종료됩니다.")
    while True:
        user_input = input("당신: ")
        if user_input.lower() == '종료':
            break
        bot_response = chat_with_bot(user_input)
        print("챗봇:", bot_response)