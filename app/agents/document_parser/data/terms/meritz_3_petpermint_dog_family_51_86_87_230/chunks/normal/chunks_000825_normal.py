from langchain_core.documents import Document

chunk = Document(
    page_content=('다) “심한 뇌전증 발작”이라 함은 월 8회 이상의 중 증발작이 연 6개월 이상의 기간에 걸쳐 발생하 고, 발작할 때 유발된 호흡장애, '
 '흡인성 폐렴, 심한 탈진, 구역질, 두통, 인지장해 등으로 요 양관리가 필요한 상태를 말한다. 라) “뚜렷한 뇌전증 발작”이라 함은 월 '
 '5회 이상의 중증발작 또는 월 10회 이상의 경증발작이 연 6 개월 이상의 기간에 걸쳐 발생하는 상태를 말한 다. 마) “약간의 뇌전증 '
 '발작”이라 함은 월 1회 이상의 중증발작 또는 월 2회 이상의 경증발작이 연 6개 월 이상의 기간에 걸쳐 발생하는 상태를 말한다'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 229},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['head']},
 'indexing': {'chunk_id': 'chunk_000825',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
