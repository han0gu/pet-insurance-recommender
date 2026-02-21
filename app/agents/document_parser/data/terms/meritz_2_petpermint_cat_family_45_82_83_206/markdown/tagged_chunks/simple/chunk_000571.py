from langchain_core.documents import Document

chunk = Document(
    page_content=('- 3항의 정상인의 신체 각 관절에 대한 평균 운동\n'
 '- 가능영역을 기준으로 정상각도 및 측정방법 등을\n'
 '- 따른다.\n'
 '- 나) 관절기능장해를 표시할 경우 장해부위의 장해각\n'
 '- 도와 정상부위의 측정치를 동시에 판단하여 장해\n'
 '- 상태를 명확히 한다. 단, 관절기능장해가 신경손\n'
 '- 상으로 인한 경우에는 운동범위 측정이 아닌 근\n'
 '- 력 및 근전도 검사를 기준으로 평가한다.\n'
 '7) “관절 하나의 기능을 완전히 잃었을 때”라 함은 아\n'
 '래의 경우 중 하나에 해당하는 경우를 말한다.- 가) 완전 강직(관절굳음)'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other',
            'risk_domains': ['digestive', 'head', 'joint']},
 'indexing': {'chunk_id': 'chunk_000571',
              'chunk_char_len': 277,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
