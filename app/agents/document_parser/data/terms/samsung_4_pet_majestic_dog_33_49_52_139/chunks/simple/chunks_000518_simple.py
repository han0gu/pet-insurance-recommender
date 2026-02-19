from langchain_core.documents import Document

chunk = Document(
    page_content=('여 의료법 제3조(의료기관)에서 규정한 국내의 병원 또는 의원에서 의사의 관리 하에 [별표-상해및질병관련3]급여 '
 '창상봉합술(안면부,단순봉합제외) 대상 수가코드에서 정 한 진료행위로 치료를 받은 경우를 말합니다.'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 96},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000518',
              'chunk_char_len': 117,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
