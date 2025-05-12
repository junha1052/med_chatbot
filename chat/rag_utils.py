from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from typing import List, Dict, Tuple, Optional
from langchain.docstore.document import Document
from langchain.vectorstores import Chroma
import google.generativeai as genai
from langchain.chains import RetrievalQA
import os, requests
from pathlib import Path
from dotenv import load_dotenv


# .env 파일 읽어오기
load_dotenv()  

API_KEY = os.getenv("GOOGLE_API_KEY")
MED_API_URL    = os.getenv("MED_API_URL")
MED_API_TOKEN  = os.getenv("MED_API_TOKEN")

genai.configure(api_key=API_KEY)

gcp_creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if gcp_creds:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gcp_creds

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


    
# ── C. 간단 키워드 검색 함수 ────────────────────────────────────────────────────
def search_medications(query: str, medications: list[dict]) -> dict:
    '''
    역할: 사용자 질문(query)에서 언급된 약물명을 all_med 리스트에서 찾아 반환
    입력: query: str — 사용자 질문
          all_med: List[Dict] — 전체 약물 정보
    출력: med: Dict — 매칭된 약물 정보 (없으면 {})
    '''
    if not isinstance(query, str) or not isinstance(medications, list):
        return {}
        
    q = query.lower()
    for med in medications:
        if not isinstance(med, dict):
            continue
        name = med.get("name", "").lower()
        if name and name in q:
            return med
    return {}

# ── D. 프롬프트 생성 함수 ──────────────────────────────────────────────────────
def create_med_context(med: dict) -> str:
    '''
    역할: search_medications가 찾아준 단일 med dict를, RAG 검색 컨텍스트(텍스트 블록)로 변환
    입력: med: Dict — 단일 약물 정보 사전
    출력:contexts: str — LLM 프롬프트에 넣을 컨텍스트 문자열
    '''
    if not med:
        return "죄송하지만, 해당 약물 정보를 찾을 수 없습니다."

    # allergy_warnings, condition_warnings 안의 dict에서 'term' 추출
    def extract_terms(warnings):
        if not warnings:
            return "없음"
        terms = []
        for w in warnings:
            if isinstance(w, dict) and "term" in w:
                terms.append(str(w["term"]))
            else:
                terms.append(str(w))
        return ", ".join(terms)

    allergy = extract_terms(med.get("allergy_warnings", []))
    condition = extract_terms(med.get("condition_warnings", []))
    side_fx = med.get("side_effects") or "없음"

    return (
        f"약물명: {med.get('name','')}\n"
        f"제조사: {med.get('manufacturer','')}\n"
        f"용량·빈도: {med.get('dosage','')}, 하루 {med.get('frequency_per_day','')}회\n"
        f"효능: {med.get('efficacy','')}\n"
        f"용법: {med.get('usage','')}\n"
        f"주의사항: {med.get('precautions','')}\n"
        f"상호작용: {med.get('interaction','')}\n"
        f"부작용: {side_fx}\n"
        f"보관: {med.get('storage','')}\n"
        f"알레르기 경고: {allergy}\n"
        f"조건별 경고: {condition}"
    )
# ── E. RAG 시스템 초기화 ─────────────────────────────────────────────────────   
def initialize_rag():
    '''
    역할: RAG 시스템 초기화
    출력: retriever - 벡터 검색기
    '''
    try:
        # 임베딩 모델 세팅 (Retrieval)
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-exp-03-07",
            api_key=API_KEY
        )
        # 문서 색인
        docs = [
            Document(page_content=create_med_context(m), metadata={"name": m["name"]})
            for m in all_med]
        # 벡터 DB 설정
        vectordb = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory="./med_chroma_db"
        )
        # vectordb.persist()
        # 리트리버 설정
        return vectordb.as_retriever(search_kwargs={"k": 3})
    except Exception as e:
        print(f"RAG 초기화 중 오류 발생: {e}")
        raise

# ── F. LLM 모델 초기화 ─────────────────────────────────────────────────────
def create_llm():
    print("GOOGLE_API_KEY", API_KEY)
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        api_key=API_KEY
    )

# ── G. 질문 분석 및 정보 결여 감지 ─────────────────────────────────────────────────────
def analyze_query(query: str, med_info: dict) -> Tuple[bool, List[str]]:
    '''
    역할: 질문을 분석하고 데이터베이스에서 찾을 수 없는 정보를 식별
    입력: query - 사용자 질문, med_info - 약물 정보
    출력: (need_rag, missing_aspects) - RAG 필요 여부와 부족한 정보 목록
    '''
    # med_info가 딕셔너리가 아니면 빈 딕셔너리로 처리
    if not isinstance(med_info, dict):
        med_info = {}
        
    # 약물 정보가 없으면 RAG 필요
    if not med_info:
        return True, ["약물 기본 정보"]
    
    # 질문에서 정보 측면을 추출
    aspects = {
        "효능": ["효능", "효과", "치료", "작용"],
        "용법": ["용법", "복용법", "복용 방법", "먹는 법", "사용법"],
        "주의사항": ["주의", "주의점", "주의사항", "조심"],
        "상호작용": ["상호작용", "함께", "같이", "병용", "다른 약", "약물"],
        "부작용": ["부작용", "이상반응", "역효과", "문제", "증상"],
        "알레르기": ["알레르기", "과민반응"],
        "보관": ["보관", "저장", "유효기간", "유통기한"],
        "음주": ["술", "알코올", "음주"],
        "운전": ["운전", "기계", "조작", "취급"],
        "임신": ["임신", "수유", "모유", "태아", "출산"],
        "어린이": ["어린이", "소아", "아기", "어린이용", "유아"]
    }
    
    # 질문에서 중요 측면 찾기
    query_lower = query.lower()
    missing_aspects = []
    need_rag = False
    
    # aspects가 딕셔너리가 아닌 경우 처리
    if not isinstance(aspects, dict):
        return True, ["약물 기본 정보"]
    
    for aspect, keywords in aspects.items():
        if not isinstance(keywords, list):
            continue
        for keyword in keywords:
            if keyword in query_lower:
                # 해당 측면에 대한 정보가 불충분한지 확인
                if aspect == "음주" and "상호작용" in med_info and "술" not in med_info["상호작용"].lower():
                    missing_aspects.append(f"{aspect} 관련 정보")
                    need_rag = True
                elif aspect == "운전" and "주의사항" in med_info and "운전" not in med_info["주의사항"].lower():
                    missing_aspects.append(f"{aspect} 관련 정보")
                    need_rag = True
                elif aspect in ["임신", "어린이"] and aspect not in str(med_info).lower():
                    missing_aspects.append(f"{aspect} 관련 정보")
                    need_rag = True
                elif aspect in med_info and (not med_info[aspect] or med_info[aspect] == "없음"):
                    missing_aspects.append(f"{aspect} 상세 정보")
                    need_rag = True
                break
    
    return need_rag, missing_aspects

# ── H. 하이브리드 응답 생성 시스템 ─────────────────────────────────────────────────────
def hybrid_answer(query: str, all_med: list[dict], retriever, llm) -> str:
    """
    데이터베이스 정보를 우선 사용하고, 필요한 경우에만 RAG로 보완
    """
    # 1. 기본 데이터베이스 검색
    med = search_medications(query, all_med)
    
    # med가 딕셔너리가 아니면 빈 딕셔너리로 처리
    if not isinstance(med, dict):
        med = {}
    
    # 2. 질문 분석하여 누락된 정보 확인
    need_rag, missing_aspects = analyze_query(query, med)
    
    # 3. 응답 전략 결정
    if not need_rag:
        # 데이터베이스만으로 충분한 경우
        contexts = create_med_context(med)
        prompt = create_prompt_with_context(query, contexts)
        return humanize_response(generate_response(prompt), query)
    else:
        # RAG로 보완이 필요한 경우
        rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever
        )
        
        # 일반적인 질문이면 RAG만 사용
        if not med:
            rag_response = rag_chain.invoke(query)
            return humanize_response(rag_response, query)
        
        # 데이터베이스 정보 + RAG 정보 결합
        db_context = create_med_context(med)
        rag_response = rag_chain.invoke(f"{query} - 특히 {', '.join(missing_aspects)}에 대해 자세히 알려주세요.")
        
        # 결합 프롬프트 생성
        combined_prompt = f"""
다음은 약물 정보 두 가지를 바탕으로 포괄적이고 정확한 답변을 생성하기 위한 프롬프트입니다.

<데이터베이스 정보>
{db_context}
</데이터베이스 정보>

<추가 검색 정보>
{rag_response}
</추가 검색 정보>

사용자 질문: {query}

위 두 정보 소스를 통합하여 사용자의 질문에 대해 친절하고 정확하며 포괄적인 답변을 제공하세요.
특히 기본 데이터베이스에 없는 {', '.join(missing_aspects)}에 대한 추가 정보를 포함해 답변하세요.
상충되는 정보가 있다면 데이터베이스 정보를 우선시하되, 추가 검색 정보로 보완하세요.
모든 중요한 정보를 누락 없이 포함하고, 중복은 피하세요.
"""
        response = generate_response(combined_prompt)
        return humanize_response(response, query)


# ── I. 프롬프트 생성 및 LLM 응답 함수 ─────────────────────────────────────────────────────
def create_prompt_with_context(query, contexts):
    '''
    역할: 컨텍스트(contexts)와 사용자 질문(query)을 합쳐, LLM에 보내는 최종 프롬프트를 만듭니다.
    입력: query: str — 사용자 질문
          contexts: str — create_med_context가 만든 텍스트 블록
    출력: prompt: str — LLM에 보낼 완성된 문자열
    '''
    template = """
다음은 검색된 약물 정보(컨텍스트)와 질문을 바탕으로
정확하고 알기 쉽게 답변을 생성하기 위한 프롬프트입니다.

<문맥>
{contexts}
</문맥>

질문: {query}

답변:
"""
    return template.format(contexts=contexts, query=query)

def generate_response(prompt: str,
                      model_name: str = "gemini-2.0-flash",
                      max_tokens: int = 512,
                      temperature: float = 0.7) -> str:
    '''
    역할: 프롬프트에 대한 LLM 응답 생성
    '''
    model = genai.GenerativeModel(model_name)
    res = model.generate_content(
        contents=prompt,
        generation_config={
            "max_output_tokens": max_tokens,
            "temperature": temperature
        }
    )
    return res.text.strip()

# ── J. 응답 더 자연스럽게 만들기 (gemini 한 번 더 감싸기기) ─────────────────────────────────────────────────────
def humanize_response(response: str, query: str) -> str:
    '''
    역할: RAG 응답을 더 친근하고 자연스럽게 가공
    입력: response - 원본 응답, query - 원본 질문
    출력: 더 자연스럽고 친절한 응답
    '''
    humanize_prompt = f"""
당신은 친절한 약사입니다. 다음 약물 정보 답변을 더 자연스럽고 친근하게 바꿔주세요:

<원본 답변>
{response}
</원본 답변>

사용자 질문: {query}

다음 지침을 따라주세요:
1. 내용의 정확성은 100% 유지하세요.
2. 딱딱한 기술적/의학적 용어는 일상적인 표현으로 바꾸세요.
3. 약사가 환자에게 친절하게 설명하는 것처럼 부드러운 어조를 사용하세요.
4. 너무 길거나 반복적인 내용은 간결하게 다듬으세요.
5. "~입니다", "~합니다" 대신 "~이에요", "~해요" 같은 친근한 종결어미를 사용하세요.
6. 의학 정보의 정확성은 반드시 유지하세요.
7. 약사나 의사에게 상담하라는 말은 꼭! 필요할 때만 말하세요.
8. 사용자가 약에 대해 질문 안 하면 약에 대해 말하지마세요.
친절하고 자연스러운 답변:"""

    return generate_response(humanize_prompt, max_tokens=512, temperature=0.5)
def answering(query: str, all_med: list[dict]):
        # 1. 기본 데이터베이스 검색
    med = search_medications(query, all_med)
    
    # med가 딕셔너리가 아니면 빈 딕셔너리로 처리
    if not isinstance(med, dict):
        med = {}
    # 데이터베이스만으로 충분한 경우
    contexts = create_med_context(med)
    prompt = create_prompt_with_context(query, contexts)
    return humanize_response(generate_response(prompt), query)