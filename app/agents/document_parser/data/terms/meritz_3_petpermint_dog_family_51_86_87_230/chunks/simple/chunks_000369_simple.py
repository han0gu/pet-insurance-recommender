from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 제2항에 정하는 조치에 다른 진료를 병행하여 실시한 경 우, 제2항에 정하는 조치(마취 비용을 포함합니다.)에 대해 서는 '
 '보험금을 지급하지 않습니다.\n'
 '제3조(수술의 정의와 장소)\n'
 '\uf000 이 계약에 있어서「수술」이라 함은 수의사가 치료가 필 요하다고 인정한 경우로서 수의사의 관리하에 치료를 직접 적인 목적으로 '
 '기구를 사용하여 생체(生體)에 절단, 절제'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 126},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000369',
              'chunk_char_len': 196,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
