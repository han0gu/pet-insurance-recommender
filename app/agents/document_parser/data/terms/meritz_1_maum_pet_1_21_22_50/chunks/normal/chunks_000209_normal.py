from langchain_core.documents import Document

chunk = Document(
    page_content=('② 단체 구성원의 일부만을 대상으로 가입하는 경우에는 대상단체의 위험과 피보험단체의 위험의 동질성이 유지되어야 합니다.\n'
 '제4조(보험의 목적의 증가 감소 또는 교체)'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 38},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000209',
              'chunk_char_len': 91,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
