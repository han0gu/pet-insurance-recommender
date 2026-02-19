from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 계약자 또는 피보험자가 알리지 않은 경우 회사가 알고 있는 최종의 주소 또는 연락처로 등기우편 등 우편물에 대한 기록이 남는 '
 '방법으로 회사가 알린 사항은 일반적으로 도달에 필요한 기간이 지난 때에 계약자 또는 피보험자에게 도달된 것으로 봅니다.'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 35,
         'page': 10},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000039',
              'chunk_char_len': 142,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
