from langchain_core.documents import Document

chunk = Document(
    page_content=('. ➅ 반려동물이 제1항의 질병 또는 상해로 치료를 받던 중에 보험기간이 만료된 경우에도 만료일부터 180일 이내의 치료비는 제2항에 '
 '따라 보상하여 드립니다. 다만, 사고일 또 는 발병일부터 365일 이내인 경우에 한합니다. ‡ 제1항의 「수술」이라 함은 수의사가 치료가 '
 '필요하다고 인정한 경우로서 수의사의 관 리하에 치료를 직접적인 목적으로 기구를 사용하여 생체(生體)에 절단, 절제 등의 조작을 가 하는 '
 '것을 말합니다. 단, 흡인, 천자 등의 조치, 신경(神經)차단(NERVE BLOCK), 미용성형'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 3},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['head']},
 'indexing': {'chunk_id': 'chunk_000018',
              'chunk_char_len': 281,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
