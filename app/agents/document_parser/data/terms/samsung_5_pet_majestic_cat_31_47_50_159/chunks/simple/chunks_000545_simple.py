from langchain_core.documents import Document

chunk = Document(
    page_content='<용어풀이>\n[핵연료물질]\n사용된 연료를 포함합니다. [핵연료물질에 의하여 오염된 물질]\n원자핵 분열 생성물을 포함합니다.',
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 98},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000545',
              'chunk_char_len': 68,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
