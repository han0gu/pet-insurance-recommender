from langchain_core.documents import Document

chunk = Document(
    page_content=("폐지 또는 변경되</p><br><p id='1' data-category='list' style='font-size:14px'>어 보험금 "
 '지급사유에 대해 판정이 불가능한 경우 폐지 또는 변경 직전의 관련 법<br>령에서 정한 기준을 따릅니다.<br>\uf000 제1항에도 '
 '불구하고, "건강보험 행위 급여․비급여 목록 및 급여 상대가치점수" 개<br>정으로 급여 판정이 변경되더라도 제1조(보험금의 지급사유) '
 '제1항의 지급사유<br>발생 당시의 "건강보험 행위 급여․비급여 목록 및 급여 상대가치점수"에 따라 이<br>미 보험금 지급여부가 판단된'),
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
 'indexing': {'chunk_id': 'chunk_000538',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
