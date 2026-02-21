from langchain_core.documents import Document

chunk = Document(
    page_content=('기간에 걸쳐 발생하는 상태를 말한다.<br>바) “중증발작”이라 함은 전신경련을 동반하는 발<br>작으로써 신체의 균형을 유지하지 못하고 '
 '쓰러지<br>는 발작 또는 의식장해가 3분이상 지속되는 발작<br>을 말한다.<br>사) “경증발작”이라 함은 운동장해가 발생하나 '
 '스스<br>로 신체의 균형을 유지할 수 있는 발작 또는 3분<br>이내에 정상으로 회복되는 발작을 말한다.</p><footer '
 "id='55' style='font-size:14px'>204</footer><p id='56'"),
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
 'indexing': {'chunk_id': 'chunk_001106',
              'chunk_char_len': 274,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
