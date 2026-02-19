from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다) "심한 뇌전증 발작" 이라 함은 월 8회 이상의 중증발작이 연 6개월 이상의 기간에 걸쳐 발생하고, 발작할 때 유발된 호흡장애, '
 '흡인성 폐렴, 심한 탈 진, 구역질, 두통, 인지장해 등으로 요양관리가 필요한 상태를 말한다. 라) "뚜렷한 뇌전증 발작" 이라 함은 월 '
 '5회 이상의 중증발작 또는 월 10회 이 상의 경증발작이 연 6개월 이상의 기간에 걸쳐 발생하는 상태를 말한다. 마) "약간의 뇌전증 '
 '발작" 이라 함은 월 1회 이상의 중증발작 또는 월 2회 이'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 148},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['head']},
 'indexing': {'chunk_id': 'chunk_000984',
              'chunk_char_len': 262,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
