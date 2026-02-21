from langchain_core.documents import Document

chunk = Document(
    page_content=('가 필요하다고 인정한 경우로서 수의사의 관리하에 치료를\n'
 '직접적인 목적으로 기구를 사용하여 생체(生體)에 절단, 절\n'
 '제 등의 조작을 가하는 것을 말합니다. 단, 흡인, 천자 등\n'
 '의 조치, 신경(神經)차단(NERVE BLOCK), 미용성형 목적의\n'
 '수술, 피임목적의 수술 및 검사, 진단을 위한 수술(생검,\n'
 '복강경검사 등)은 제외합니다.158【용어의 정의】- - 절단(切斷): 특정부위를 잘라 내는 것\n'
 '- - 절제(切除): 특정부위를 잘라 없애는 것\n'
 '- - 흡인(吸引): 주사기 등으로 빨아들이는 것'),
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
 'indexing': {'chunk_id': 'chunk_000419',
              'chunk_char_len': 276,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
