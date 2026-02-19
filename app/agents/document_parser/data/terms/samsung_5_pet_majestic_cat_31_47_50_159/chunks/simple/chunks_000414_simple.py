from langchain_core.documents import Document

chunk = Document(
    page_content=('사」 라 합니다)에 의하여 5대골절로 치료가 필요하다고 인정된 경우로서, 자택 등에서 의 치료가 곤란하여 의료법 제3조(의료기관)에서 '
 '규정한 국내의 병원, 의원 또는 국외 의 의료관련법에서 정한 의료기관에서 의사의 관리 하에 5대골절의 치료를 직접적인 목적으로 의료기구를 '
 '사용하여 생체(生體)에 절단(切断, 특정부위를 잘라내는 것), 절 제(切除, 특정부위를 잘라 없애는 것) 등의 조작(操作)을 가하는 것을 '
 '말합니다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 77},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['joint', 'other']},
 'indexing': {'chunk_id': 'chunk_000414',
              'chunk_char_len': 234,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
