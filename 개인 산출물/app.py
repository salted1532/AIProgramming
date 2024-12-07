from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from openai.types.chat import ChatCompletionMessageParam
import openai

# 환경 변수 로드
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

class DocumentQABot:
    def __init__(self, model_name="gpt-3.5-turbo", persist_directory='chroma_store'):
        self.model_name = model_name
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings
        )

    def build_prompt(self, query, context):
        system_message: ChatCompletionMessageParam = {
            "role": "system",
            "content": """다음 제공된 컨텍스트만을 기반으로 질문에 답변해주세요.
            컨텍스트에 충분한 정보가 없다면 '확실하지 않습니다'라고 말한 후 추측해주세요.
            답변은 읽기 쉽게 단락으로 나누어 주세요."""
        }
        
        user_message: ChatCompletionMessageParam = {
            "role": "user",
            "content": f"질문: {query}\n\n제공된 컨텍스트:\n{context}"
        }
        
        return [system_message, user_message]

    def get_response(self, query, context):
        try:
            response = openai.chat.completions.create(
                model=self.model_name,
                messages=self.build_prompt(query, context)
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"오류가 발생했습니다: {str(e)}"

    def search_documents(self, query, k=5):
        results = self.vectorstore.similarity_search(
            query=query,
            k=k
        )
        return [doc.page_content for doc in results]

# 봇 인스턴스 생성
bot = DocumentQABot()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    query = data.get('question', '')
    
    if not query:
        return jsonify({'error': '질문을 입력해주세요.'})

    # 문서 검색
    relevant_docs = bot.search_documents(query)
    
    if not relevant_docs:
        return jsonify({'answer': '관련 문서를 찾을 수 없습니다.'})

    # 답변 생성
    context = " ".join(relevant_docs)
    response = bot.get_response(query, context)
    
    return jsonify({'answer': response})

if __name__ == '__main__':
    app.run(debug=True) 