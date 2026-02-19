from langchain_core.documents import Document

chunk = Document(
    page_content=('. 8) 발가락 관절의 운동범위 측정은 장해평가시점의 ｢산업 재해보상보험법 시행규칙｣ 제47조 제1항 및 제3항의 정 상인의 신체 각 '
 '관절에 대한 평균 운동가능영역을 기준 으로 정상각도 및 측정방법 등을 따른다.'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 198},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive', 'joint']},
 'indexing': {'chunk_id': 'chunk_000722',
              'chunk_char_len': 118,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
