from langchain_core.documents import Document

chunk = Document(
    page_content=('예) 자가용에서 영업용으로 변경, 영업용에서 자가용으로 변경 등 3. 보험증권 등에 기재된 피보험자의 운전여부가 변경된 경우 예) '
 '비운전자에서 운전자로 변경, 운전자에서 비운전자로 변경 등 4'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 37},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000062',
              'chunk_char_len': 107,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
