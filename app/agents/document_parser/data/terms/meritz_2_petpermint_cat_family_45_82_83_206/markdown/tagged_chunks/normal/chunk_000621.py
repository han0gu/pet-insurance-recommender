from langchain_core.documents import Document

chunk = Document(
    page_content=('- 바) “중증발작”이라 함은 전신경련을 동반하는 발\n'
 '- 작으로써 신체의 균형을 유지하지 못하고 쓰러지\n'
 '- 는 발작 또는 의식장해가 3분이상 지속되는 발작\n'
 '- 을 말한다.\n'
 '- 사) “경증발작”이라 함은 운동장해가 발생하나 스스\n'
 '- 로 신체의 균형을 유지할 수 있는 발작 또는 3분\n'
 '- 이내에 정상으로 회복되는 발작을 말한다.\n'
 '204<붙임># 일상생활 기본동작(ADLs) 제한 장해평가표| 유형 | 제한정도에 따른 지급률 |\n'
 '| --- | --- |'),
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
 'indexing': {'chunk_id': 'chunk_000621',
              'chunk_char_len': 250,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
