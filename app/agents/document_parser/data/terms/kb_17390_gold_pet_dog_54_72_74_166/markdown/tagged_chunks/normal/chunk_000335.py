from langchain_core.documents import Document

chunk = Document(
    page_content=('- \uf000 제1항에도 불구하고, 보건복지부에서 고시하는「건강보험 행위 급여․비급여 목록 약\n'
 '반KB 금쪽같은 펫보험(강아지)(무배당)(26.01) 83- 83 -상해질병- \n'
 '- \n'
 '및 급여 상대가치점수」의 개정에 따라 제1항의 "수가코드"가 폐지 또는 변경되- 어 보험금 지급사유에 대해 판정이 불가능한 경우 폐지 '
 '또는 변경 직전의 관련 법\n'
 '- 령에서 정한 기준을 따릅니다.\n'
 '- \uf000 제1항에도 불구하고, "건강보험 행위 급여․비급여 목록 및 급여 상대가치점수" 개'),
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
 'indexing': {'chunk_id': 'chunk_000335',
              'chunk_char_len': 255,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
