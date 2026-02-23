from langchain_core.documents import Document

chunk = Document(
    page_content=('계약을 체결하는 데 필요한 사항을 질문 또는 설명하는 방법. 이 경우 계약자의 답변과 확인내\n'
 '용을 음성 녹음함으로써 약관의 중요한 내용을 설명한 것으로 봅니다.【통신판매계약】 전화 · 우편 · 인터넷 등 통신수단을 이용하여 '
 '체결하는 계약을 말합니다.③ 회사가 제1항에 따라 제공될 약관 및 계약자 보관용 청약서를 청약할 때 계약자에게 전달하지 않\n'
 '거나 약관의 중요한 내용을 설명하지 않은 때 또는 계약을 체결할 때 계약자가 청약서에 자필서명'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000048',
              'chunk_char_len': 246,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
