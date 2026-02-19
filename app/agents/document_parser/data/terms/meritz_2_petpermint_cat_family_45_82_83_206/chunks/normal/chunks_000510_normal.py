from langchain_core.documents import Document

chunk = Document(
    page_content=('제3조(수술의 정의와 장소)\n'
 '\uf000 이 특별약관에 있어서「수술」이라 함은 수의사가 치료 가 필요하다고 인정한 경우로서 수의사의 관리하에 치료를 직접적인 목적으로 '
 '기구를 사용하여 생체(生體)에 절단, 절 제 등의 조작을 가하는 것을 말합니다. 단, 흡인, 천자 등 의 조치, '
 '신경(神經)차단(NERVE BLOCK), 미용성형 목적의 수술, 피임목적의 수술 및 검사, 진단을 위한 수술(생검, 복강경검사 등)은 '
 '제외합니다.\n'
 '【용어의 정의】'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 151},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive', 'head']},
 'indexing': {'chunk_id': 'chunk_000510',
              'chunk_char_len': 241,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
