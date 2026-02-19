from langchain_core.documents import Document

chunk = Document(
    page_content=('제2조(계약자)\n'
 '이 특별약관의 계약자는 제1조(적용범위)의 단체를 대표하여 계약상의 모든 권리, 의무를 행사할 수 있어야 합니다.\n'
 '제3조(보험가입금액)\n'
 '피보험자의 보험가입금액은 동일하게 책정하는 것을 원칙으로 합니다.\n'
 '제4조(피보험자의 증가, 감소 또는 교체)'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 35,
         'page': 32},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000154',
              'chunk_char_len': 145,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.9}},
)
