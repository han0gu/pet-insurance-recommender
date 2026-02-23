from langchain_core.documents import Document

chunk = Document(
    page_content=('발작할 때 유발된 호흡장애, 흡인성 폐렴,<br>심한 탈진, 구역질, 두통, 인지장해 등으로 요<br>양관리가 필요한 상태를 '
 '말한다.<br>라) “뚜렷한 뇌전증 발작”이라 함은 월 5회 이상의<br>중증발작 또는 월 10회 이상의 경증발작이 연 6<br>개월 '
 '이상의 기간에 걸쳐 발생하는 상태를 말한<br>다.<br>마) “약간의 뇌전증 발작”이라 함은 월 1회 이상의<br>중증발작 또는 월 '
 '2회 이상의 경증발작이 연 6개<br>월 이상의 기간에 걸쳐 발생하는 상태를 말한다.<br>바) “중증발작”이라 함은 전신경련을 동반하는'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'head']},
 'indexing': {'chunk_id': 'chunk_001105',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
