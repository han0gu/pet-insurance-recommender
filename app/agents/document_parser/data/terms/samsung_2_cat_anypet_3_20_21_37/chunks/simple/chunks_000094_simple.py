from langchain_core.documents import Document

chunk = Document(
    page_content=('2. 계약자 또는 피보험자의 책임 있는 사유에 의하는 경우 : 이미 경과한 기간에 대하여 단기요율 로 계산한 보험료를 뺀 잔액. 다만, '
 '계약자, 피보험자의 고의 또는 중대한 과실로 무효가 된 때 에는 보험료를 돌려드리지 않습니다.'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 35,
         'page': 18},
 'term_type': 'basic',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000094',
              'chunk_char_len': 128,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
