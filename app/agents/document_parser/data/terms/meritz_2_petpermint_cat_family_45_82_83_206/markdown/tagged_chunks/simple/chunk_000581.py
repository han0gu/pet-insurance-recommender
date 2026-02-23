from langchain_core.documents import Document

chunk = Document(
    page_content=('평가한다.- 가) 각 관절의 운동범위 측정은 장해평가시점의 ｢산\n'
 '- 업재해보상보험법 시행규칙｣ 제47조 제1항 및 제\n'
 '- 3항의 정상인의 신체 각 관절에 대한 평균 운동\n'
 '- 가능영역을 기준으로 정상각도 및 측정방법 등을\n'
 '- 따른다.\n'
 '- 나) 관절기능장해가 신경손상으로 인한 경우에는 운\n'
 '- 동범위 측정이 아닌 근력 및 근전도 검사를 기준\n'
 '- 으로 평가한다.\n'
 '7) “관절 하나의 기능을 완전히 잃었을 때”라 함은 아\n'
 '래의 경우 중 하나에 해당하는 때를 말한다.- 가) 완전 강직(관절굳음)'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage',
            'risk_domains': ['digestive', 'head', 'joint']},
 'indexing': {'chunk_id': 'chunk_000581',
              'chunk_char_len': 272,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
