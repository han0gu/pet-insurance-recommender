from langchain_core.documents import Document

chunk = Document(
    page_content=('- 이상의 기간에 걸쳐 발생하고, 발작할 때 유발된 호흡장애, 흡인성\n'
 '- 폐렴, 심한 탈진, 구역질, 두통, 인지장해 등으로 요양관리가 필요\n'
 '- 한 상태를 말한다.\n'
 '- 라) “뚜렷한 뇌전증 발작”이라 함은 월 5회 이상의 중증발작 또는 월\n'
 '- 10회 이상의 경증발작이 연 6개월 이상의 기간에 걸쳐 발생하는 상\n'
 '- 태를 말한다.\n'
 '- 마) “약간의 뇌전증 발작”이라 함은 월 1회 이상의 중증발작 또는 월 2\n'
 '- 회 이상의 경증발작이 연 6개월 이상의 기간에 걸쳐 발생하는 상태\n'
 '- 를 말한다.'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'head']},
 'indexing': {'chunk_id': 'chunk_000943',
              'chunk_char_len': 275,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
