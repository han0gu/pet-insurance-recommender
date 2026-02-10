from langchain_core.documents import Document

def get_mock_rag_data():
    """RAG 팀이 아직이라 임시로 만든 가짜 약관 데이터"""
    return [
        Document(
            page_content="[A사 튼튼 펫보험] 8세 이하 가입 가능. 슬개골 탈구 면책(보장안함).",
            metadata={"source": "A_insurance.pdf", "score": 0.9}
        ),
        Document(
            page_content="[B사 시니어 케어] 15세까지 가입 가능. 슬개골 탈구 및 심장 질환 수술비 보장.",
            metadata={"source": "B_senior.pdf", "score": 0.85}
        ),
        Document(
            page_content="[C사 실속 보험] 전 연령 가입 가능. 수술비 미보장(통원치료만).",
            metadata={"source": "C_basic.txt", "score": 0.7}
        )
    ]