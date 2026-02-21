from langchain_core.documents import Document

chunk = Document(
    page_content=('- 9) 손가락의 관절기능장해 평가는 손가락 관절의 관절운동범위 제한 등으로\n'
 '- 평가한다. 각 관절의 운동범위 측정은 장해평가시점의 ｢산업재해보상보\n'
 '- 험법 시행규칙｣ 제47조 제1항 및 제3항의 정상인의 신체 각 관절에 대한\n'
 '- 평균 운동가능영역을 기준으로 정상각도 및 측정방법 등을 따른다.\n'
 '| 부 가 설 명 | 손가락 ![image](/image/placeholder)\n'
 ' |\n'
 '| --- | --- |\n'
 '부 가 설 명손가락![image](/image/placeholder)'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive', 'joint']},
 'indexing': {'chunk_id': 'chunk_000916',
              'chunk_char_len': 265,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
