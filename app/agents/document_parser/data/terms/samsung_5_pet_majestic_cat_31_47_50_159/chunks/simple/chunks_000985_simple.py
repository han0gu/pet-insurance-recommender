from langchain_core.documents import Document

chunk = Document(
    page_content=('상의 경증발작이 연 6개월 이상의 기간에 걸쳐 발생하는 상태를 말한다.\n'
 '바) "중증발작" 이라 함은 전신경련을 동반하는 발작으로써 신체의 균형을 유 지하지 못하고 쓰러지는 발작 또는 의식장해가 3분이상 '
 '지속되는 발작을 말한다. 사) "경증발작" 이라 함은 운동장해가 발생하나 스스로 신체의 균형을 유지할 수 있는 발작 또는 3분 이내에 '
 '정상으로 회복되는 발작을 말한다.\n'
 '<붙임 : 일상생활 기본동작(ADLs) 제한 장해평가표>\n'
 '유형 | 제한 정도에 따른 지급률'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 149},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['joint', 'head', 'other']},
 'indexing': {'chunk_id': 'chunk_000985',
              'chunk_char_len': 257,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
