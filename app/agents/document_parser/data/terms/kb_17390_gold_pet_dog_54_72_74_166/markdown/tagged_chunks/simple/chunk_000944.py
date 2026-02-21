from langchain_core.documents import Document

chunk = Document(
    page_content=('- 회 이상의 경증발작이 연 6개월 이상의 기간에 걸쳐 발생하는 상태\n'
 '- 를 말한다.\n'
 '- 바) “중증발작”이라 함은 전신경련을 동반하는 발작으로써 신체의 균\n'
 '- 형을 유지하지 못하고 쓰러지는 발작 또는 의식장해가 3분 이상 지\n'
 '- 속되는 발작을 말한다.\n'
 '- 사) “경증발작”이라 함은 운동장해가 발생하나 스스로 신체의 균형을\n'
 '- 유지할 수 있는 발작 또는 3분 이내에 정상으로 회복되는 발작을 말\n'
 '- 한다.\n'
 '<붙 임>| ∙ 일상생활 | 기본동작(ADLs) 제한 장해평가표 | 공 통 |\n'
 '| --- | --- | --- |'),
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
 'indexing': {'chunk_id': 'chunk_000944',
              'chunk_char_len': 291,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
