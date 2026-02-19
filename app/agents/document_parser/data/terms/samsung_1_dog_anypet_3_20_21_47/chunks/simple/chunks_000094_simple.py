from langchain_core.documents import Document

chunk = Document(
    page_content=('【위법 계약】 금융상품판매업자 등이 「금융소비자보호에 관한 법률」 제47조에서 정한 적합성원칙, 적 정성원칙, 설명의무, 불공정영업행위 '
 '금지 또는 부당권유행위 금지를 위반한 계약을 말합니다.'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 17},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000094',
              'chunk_char_len': 106,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
