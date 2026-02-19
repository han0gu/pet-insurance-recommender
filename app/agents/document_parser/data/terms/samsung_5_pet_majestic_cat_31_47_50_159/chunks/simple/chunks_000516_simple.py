from langchain_core.documents import Document

chunk = Document(
    page_content=('여 의료법 제3조(의료기관)에서 규정한 국내의 병원 또는 의원에서 의사의 관리 하에 [별표-상해및질병관련3]급여 '
 '창상봉합술(안면부,단순봉합제외) 대상 수가코드에서 정 한 진료행위로 치료를 받은 경우를 말합니다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 94},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000516',
              'chunk_char_len': 117,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
