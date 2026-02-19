from langchain_core.documents import Document

chunk = Document(
    page_content=('1. 전화를 이용하여 청약내용, 보험료납입, 보험기간, 계약 전 알릴 의무, 약관의 중요한 내용 등 계약을 체결하는 데 필요한 사항을 '
 '질문 또는 설명하는 방법. 이 경우 계약자의 답변과 확인내 용을 음성 녹음함으로써 약관의 중요한 내용을 설명한 것으로 봅니다.\n'
 '【통신판매계약】 전화 · 우편 · 인터넷 등 통신수단을 이용하여 체결하는 계약을 말합니다.'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 12},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000056',
              'chunk_char_len': 196,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
