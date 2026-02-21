from langchain_core.documents import Document

chunk = Document(
    page_content=('- 사」 라 합니다)에 의하여 상해흉터복원으로 치료가 필요하다고 인정된 경우로서 자택\n'
 '- 등에서의 치료가 곤란하여 의료법 제3조(의료기관)에서 규정한 국내의 병원, 의원 또\n'
 '- 는 국외의 의료관련법에서 정한 의료기관에서 의사의 관리 하에 상해흉터복원의 치료\n'
 '- 를 직접적인 목적으로 의료기구를 사용하여 생체(生體)에 절단(切断, 특정부위를 잘라\n'
 '- 내는 것), 절제(切除, 특정부위를 잘라 없애는 것) 등의 조작(操作)을 가하는 것을 말합\n'
 '- 니다.'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000360',
              'chunk_char_len': 251,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
