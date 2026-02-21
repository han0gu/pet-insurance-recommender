from langchain_core.documents import Document

chunk = Document(
    page_content=('- 목관절(완관절)을 말한다. 규정\n'
 '- 5) ‘한 팔의 손목 이상을 잃었을 때’라 함은 손목관절(완관절)부터(손목\n'
 '- 관절 포함) 심장에 가까운 쪽에서 절단된 때를 말하며, 팔꿈치관절(주관\n'
 '- 절) 상부에서 절단된 경우도 포함한다.\n'
 '- 6) 팔의 관절기능장해 평가는 팔의 3대 관절의 관절운동범위 제한 등으로\n'
 '- 평가한다.\n'
 '- 가) 각 관절의 운동범위 측정은 장해평가시점의 ｢산업재해보상보험법\n'
 '- 시행규칙｣ 제47조 제1항 및 제3항의 정상인의 신체 각 관절에 대한\n'
 '- 평균 운동가능영역을 기준으로 정상각도 및 측정방법 등을 따른다.'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'joint']},
 'indexing': {'chunk_id': 'chunk_000893',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
