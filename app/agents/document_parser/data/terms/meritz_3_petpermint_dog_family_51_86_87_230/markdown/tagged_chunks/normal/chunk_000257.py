from langchain_core.documents import Document

chunk = Document(
    page_content=('- inhibitor) 약물\n'
 '\uf000 제2항에 정하는 조치에 다른 진료를 병행하여 실시한 경\n'
 '우, 제2항에 정하는 조치(마취 비용을 포함합니다.)에 대해\n'
 '서는 보험금을 지급하지 않습니다.# 제3조(수술의 정의와 장소)\uf000 이 계약에 있어서「수술」이라 함은 수의사가 치료가 '
 '필116요하다고 인정한 경우로서 수의사의 관리하에 치료를 직접\n'
 '적인 목적으로 기구를 사용하여 생체(生體)에 절단, 절제\n'
 '등의 조작을 가하는 것을 말합니다. 단, 흡인, 천자 등의\n'
 '조치, 신경(神經)차단(NERVE BLOCK), 미용성형 목적의 수'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive', 'head']},
 'indexing': {'chunk_id': 'chunk_000257',
              'chunk_char_len': 285,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
