from langchain_core.documents import Document

chunk = Document(
    page_content=('- 제거된 후 장해를 평가한다. 단, 제거가 불가능한 경\n'
 '- 우에는 고정물 등이 있는 상태에서 장해를 평가한다.\n'
 '- 2) 관절을 사용하지 않아 발생한 일시적인 기능장해(예\n'
 '- 를 들면 캐스트로 환부를 고정시켰기 때문에 치유후\n'
 '- 의 관절에 기능장해가 발생한 경우)는 장해로 평가하\n'
 '- 지 않는다.\n'
 '- 3) “팔”이라 함은 어깨관절(견관절)부터 손목관절(완\n'
 '- 관절)까지를 말한다.\n'
 '- 4) “팔의 3대관절”이라 함은 어깨관절(견관절), 팔꿈치\n'
 '- 관절(주관절), 손목관절(완관절)을 말한다.'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'joint']},
 'indexing': {'chunk_id': 'chunk_000569',
              'chunk_char_len': 274,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
