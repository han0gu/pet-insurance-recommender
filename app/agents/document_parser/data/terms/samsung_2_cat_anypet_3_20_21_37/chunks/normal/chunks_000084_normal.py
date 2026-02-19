from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 보험설계 사 등의 행위가 없었다 하더라도 계약자 또는 피보험자가 사실대로 알리지 않거나 부실한 사항 을 알렸다고 인정되는 '
 '경우에는 계약을 해지할 수 있습니다.'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 35,
         'page': 16},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000084',
              'chunk_char_len': 94,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
