from .rag_utils import initialize_rag, create_llm, hybrid_answer
import os, requests, re
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import google.generativeai as genai
from langchain.docstore.document import Document




# ── A. 환경 변수 로드 ────────────────────────────────────────────────
API_KEY = os.getenv("GOOGLE_API_KEY")
MED_API_URL = os.getenv("MED_API_URL")
MED_API_TOKEN = os.getenv("MED_API_TOKEN")

# ── B. Google Generative AI API 초기화 ────────────────────────────────────────────────

genai.configure(api_key=API_KEY)


# ── B. 백엔드에서 약물 데이터 로드 ────────────────────────────────────────────────

'''
import json
from pathlib import Path

_txt_path =  "C://Users/peter/OneDrive/Desktop/solution/med_chatbot/chat/response.txt"


def _local_get(url, headers=None, *args, **kwargs):
    text = open(_txt_path, encoding="utf-8").read()
    # 파일에서 [ 로 시작하는 JSON 배열 부분만 추출
    m = re.search(r'(\[.*\])', text, re.DOTALL)
    if not m:
        raise RuntimeError("JSON 배열을 찾을 수 없습니다.")
    json_str = m.group(1)
    data = json.loads(json_str)
    class R:
        status_code = 200
        def raise_for_status(self): pass
        def json(self): return data
    return R()
requests.get = _local_get
'''
def load_medications() -> list[dict]:
    '''
    역할: 백엔드(API)에서 전체 약물 정보를 한 번에 불러오기
    입력: 없음 (URL과 인증 토큰은 내부에서 처리)
    출력: List[Dict] 형태의 약물 정보 리스트 (all_med)
    '''
    headers = {
    "Authorization": f"Token {MED_API_TOKEN}"}
    resp = requests.get(MED_API_URL, headers=headers)
    resp.raise_for_status()
    return resp.json()

all_med = load_medications()

class ChatBotAPIView(APIView):
    """
    POST /api/chat/bot/
    Body: {"query": "사용자 질문"}
    """

    
    def post(self, request):
        query = request.data.get('query', '')
        if not query:
            return Response({"error": "query is required"}, status=status.HTTP_400_BAD_REQUEST)
        
        # 1. 약물 정보 로드
        all_med = load_medications()
    
        # 2. RAG + LLM 초기화
        retriever = initialize_rag()
        llm       = create_llm()
        
        # 3. 하이브리드 응답 생성
        answer = hybrid_answer(query, all_med, retriever, llm)

        #answer= answering(query, all_med)

        return Response({"answer": answer})
    