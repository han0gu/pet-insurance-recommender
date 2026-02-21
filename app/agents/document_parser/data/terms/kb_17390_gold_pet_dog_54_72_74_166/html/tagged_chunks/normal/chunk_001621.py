from langchain_core.documents import Document

chunk = Document(
    page_content=('. 각 관절의 운동범위 측정은 장해평가시점의 ｢산업재해보상보<br>험법 시행규칙｣ 제47조 제1항 및 제3항의 정상인의 신체 각 관절에 '
 "대한<br>평균 운동가능영역을 기준으로 정상각도 및 측정방법 등을 따른다.</p><br><table id='137' "
 "style='font-size:20px'><thead></thead><tbody><tr><td>부 가 설 명</td><td>손가락 "
 '<figure><img alt="" data-coord="top-left:(836,121); bottom-right:(1423,481)"'),
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
 'indexing': {'chunk_id': 'chunk_001621',
              'chunk_char_len': 290,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
