from langchain_core.documents import Document

chunk = Document(
    page_content=('공하여 드립니다. 만약, 회사가 전자우편 및 전자적 의사표시로 제공한 경우 계약자 또는 그 대리 인이 약관 및 계약자 보관용 청약서 등을 '
 '수신하였을 때에는 해당 문서를 드린 것으로 봅니다.\n'
 '1. 서면교부 2. 우편 또는 전자우편 3. 휴대전화 문자메세지 또는 이에 준하는 전자적 의사표시\n'
 '【약관의 중요한 내용】 금융소비자 보호에 관한 법률 제19조(설명의무), 동법 시행령 제13조, 금융소 비자 보호에 관한 감독규정 '
 '[별표3]에서 정한 다음의 내용을 포함합니다.'),
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
 'indexing': {'chunk_id': 'chunk_000054',
              'chunk_char_len': 259,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
