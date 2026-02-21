from langchain_core.documents import Document

chunk = Document(
    page_content=('- 의사"의 관리하에 의료법 제3조(의료기관) 제2항에서 규정한 국내의 병원 및 의\n'
 '- 원에서 행한 의료행위에 한합니다.\n'
 '- \uf000 제1항에도 불구하고, 보건복지부에서 고시하는 "건강보험 행위 급여․비급여 목록\n'
 '- 및 급여 상대가치점수"의 개정에 따라 제1항의 "수가코드"가 폐지 또는 변경되어\n'
 '- 보험금 지급사유에 대해 판정이 불가능한 경우 폐지 또는 변경 직전의 관련 법령\n'
 '- 에서 정한 기준을 따릅니다.\n'
 '\uf000 제1항에도 불구하고, "건강보험 행위 급여․비급여 목록 및 급여 상대가치점수" 개'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000350',
              'chunk_char_len': 274,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
