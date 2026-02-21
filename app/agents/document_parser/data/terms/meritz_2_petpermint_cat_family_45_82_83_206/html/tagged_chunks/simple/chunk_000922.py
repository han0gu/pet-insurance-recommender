from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 각막이식술을 받은 환자인 경<br>우 각막이식술 이전의 시력상태를 기준으로 평가한다.<br>3) “한 눈이 멀었을 때”라 함은 '
 '안구의 적출은 물론<br>명암을 가리지 못하거나(“광각무”) 겨우 가릴 수<br>있는 경우(“광각유”)를 말한다.<br>4) “한눈의 '
 '교정시력이 0.02이하로 된 때”라 함은 안<br>전수동(Hand Movement)주1), '
 "안전수지(Finger<br>Counting)주2) 상태를 포함한다.</p><br><p id='8' "
 "data-category='paragraph'"),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['eye']},
 'indexing': {'chunk_id': 'chunk_000922',
              'chunk_char_len': 283,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
