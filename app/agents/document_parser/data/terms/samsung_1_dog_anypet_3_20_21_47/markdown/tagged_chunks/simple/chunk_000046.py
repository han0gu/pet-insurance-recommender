from langchain_core.documents import Document

chunk = Document(
    page_content=('- 2. 우편 또는 전자우편\n'
 '- 3. 휴대전화 문자메세지 또는 이에 준하는 전자적 의사표시\n'
 '【약관의 중요한 내용】 금융소비자 보호에 관한 법률 제19조(설명의무), 동법 시행령 제13조, 금융소\n'
 '비자 보호에 관한 감독규정 [별표3]에서 정한 다음의 내용을 포함합니다.- - 보험금 지급제한 사유 및 지급절차\n'
 '- - 청약의 철회에 관한 사항\n'
 '- - 계약의 해지 및 해제\n'
 '- - 분쟁조정 절차에 관한 사항\n'
 '- - 예금자보호법에 따른 보호여부\n'
 '- - 환급금에 관한 사항\n'
 '- - 고지의무 및 통지의무 위반의 효과'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000046',
              'chunk_char_len': 282,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
