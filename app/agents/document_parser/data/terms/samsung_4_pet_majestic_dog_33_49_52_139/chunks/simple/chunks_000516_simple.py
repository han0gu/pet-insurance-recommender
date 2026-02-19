from langchain_core.documents import Document

chunk = Document(
    page_content=('② 이 특별약관에서 「창상봉합술(급여)」 이라 함은 병원 또는 의원의 의사에 의하여 치료 가 필요하다고 인정된 경우로서 자택 등에서의 '
 '치료가 곤란하여 의료법 제3조(의료기 관)에서 규정한 국내의 병원 또는 의원에서 의사의 관리 하에 [별표-상해및질병관련 1]급여 '
 '창상봉합술 대상 수가코드에서 정한 진료행위로 치료를 받은 경우를 말합니 다'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 96},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['skin', 'other']},
 'indexing': {'chunk_id': 'chunk_000516',
              'chunk_char_len': 188,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
