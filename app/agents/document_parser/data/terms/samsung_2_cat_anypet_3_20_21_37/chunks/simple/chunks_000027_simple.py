from langchain_core.documents import Document

chunk = Document(
    page_content=('【가지급보험금】 보험금이 지급기한 내에 지급되지 못할 것으로 판단되는 경우 회사가 예상되는 보험금 의 일부를 먼저 지급하는 제도로 '
 '피보험자가 필요로 하는 비용을 보전해 주기 위해 회사가 먼저 지급하는 임시 교부금을 말합니다.'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 35,
         'page': 8},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000027',
              'chunk_char_len': 125,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
