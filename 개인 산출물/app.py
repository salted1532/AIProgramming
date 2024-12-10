from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from openai.types.chat import ChatCompletionMessageParam
from langchain.text_splitter import RecursiveCharacterTextSplitter
import openai
import tiktoken

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
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        self.tokenizer = tiktoken.encoding_for_model(model_name)

    def count_tokens(self, text):
        return len(self.tokenizer.encode(text))

    def build_prompt(self, query, context):
        system_message: ChatCompletionMessageParam = {
            "role": "system",
            "content": """다음 제공된 컨텍스트만을 기반으로 질문에 답변해주세요.
            컨텍스트에 충분한 정보가 없다면 '확실하지 않습니다'라고 말한 후 추측해주세요.
            답변은 읽기 쉽게 단락으로 나누어 주세요.
            답변 시 가능한 한 컨텍스트의 정보를 최대한 활용하여 상세하게 설명해주세요.
            답변에는 관련된 구체적인 예시나 참조를 포함시켜주세요."""
        }
        
        context_str = "\n\n".join([f"문서 {i+1}:\n{doc}" for i, doc in enumerate(context)])
        
        user_message: ChatCompletionMessageParam = {
            "role": "user",
            "content": f"질문: {query}\n\n제공된 컨텍스트:\n{context_str}"
        }

        # 토큰 수 계산
        system_tokens = self.count_tokens(system_message["content"])
        context_tokens = self.count_tokens(context_str)
        query_tokens = self.count_tokens(query)
        total_tokens = system_tokens + context_tokens + query_tokens
        
        print(f"시스템 메시지 토큰 수: {system_tokens}")
        print(f"컨텍스트 토큰 수: {context_tokens}")
        print(f"질문 토큰 수: {query_tokens}")
        print(f"총 토큰 수: {total_tokens}")
        
        return [system_message, user_message]

    def get_response(self, query, context):
        try:
            messages = self.build_prompt(query, context)
            response = openai.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.7,
                max_tokens=2000
            )
            completion_tokens = response.usage.completion_tokens
            total_tokens = response.usage.total_tokens
            print(f"응답 토큰 수: {completion_tokens}")
            print(f"전체 사용된 토큰 수: {total_tokens}")
            return response.choices[0].message.content
        except Exception as e:
            return f"오류가 발생했습니다: {str(e)}"

    def search_documents(self, query, k=5):
        # MMR(Maximum Marginal Relevance)를 사용하여 다양성 있는 문서 검색
        results = self.vectorstore.max_marginal_relevance_search(
            query=query,
            k=k,
            fetch_k=k*2,  # 더 많은 후보 문서 검색
            lambda_mult=0.7  # 다양성과 관련성의 균형을 조절
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
    response = bot.get_response(query, relevant_docs)
    
    return jsonify({'answer': response})

if __name__ == '__main__':
    app.run(debug=True)