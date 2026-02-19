from langchain_core.documents import Document

chunk = Document(
    page_content=('. (예시) 원금 : 100원, 이자율 : 연 10% 1년 후 단리계산법 : 원금 + (원금×10%) = 110원 복리계산법 : 원금 + '
 '(원금×10%) = 110원 2년 후 단리계산법 : 원금 + (원금×10%) + (원금×10%) = 120원 복리계산법 : 원금 + '
 '(원금×10%) + [원금 + (원금×10%)] ×10% = 121원'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 5},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000009',
              'chunk_char_len': 189,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
