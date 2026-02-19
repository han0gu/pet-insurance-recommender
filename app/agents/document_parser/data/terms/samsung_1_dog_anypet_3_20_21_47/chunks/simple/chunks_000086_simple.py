from langchain_core.documents import Document

chunk = Document(
    page_content=('2. 뚜렷한 위험의 변경 또는 증가와 관련된 제13조(계약 후 알릴 의무)에서 정한 계약 후 알릴 의 무를 계약자, 피보험자 또는 이들의 '
 '대리인이 이행하지 않았을 때\n'
 '④ 제3항 제1호의 경우에도 불구하고 다음 중 하나에 해당하는 경우에는 회사는 계약을 해지할 수 없 습니다.'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 16},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000086',
              'chunk_char_len': 153,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
