from langchain_core.documents import Document

chunk = Document(
    page_content=('가) 각 관절의 운동범위 측정은 장해평가시점의 ｢산 업재해보상보험법 시행규칙｣ 제47조 제1항 및 제 3항의 정상인의 신체 각 관절에 '
 '대한 평균 운동 가능영역을 기준으로 정상각도 및 측정방법 등을 따른다. 나) 관절기능장해를 표시할 경우 장해부위의 장해각 도와 정상부위의 '
 '측정치를 동시에 판단하여 장해 상태를 명확히 한다. 단, 관절기능장해가 신경손 상으로 인한 경우에는 운동범위 측정이 아닌 근 력 및 '
 '근전도 검사를 기준으로 평가한다.'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 191},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['joint']},
 'indexing': {'chunk_id': 'chunk_000690',
              'chunk_char_len': 244,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
