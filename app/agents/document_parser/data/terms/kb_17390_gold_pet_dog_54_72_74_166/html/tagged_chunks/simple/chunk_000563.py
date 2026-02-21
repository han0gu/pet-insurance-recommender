from langchain_core.documents import Document

chunk = Document(
    page_content=("변경 직전의 관련 법령<br>에서 정한 기준을 따릅니다.</p><br><p id='38' data-category='paragraph' "
 'style=\'font-size:14px\'>\uf000 제1항에도 불구하고, "건강보험 행위 급여․비급여 목록 및 급여 상대가치점수" '
 "개<br>정으로 급여 판정이 변경되더라도 제1조(보험금의 지급사유)의 지급사유 발생 당<br>특</p><br><p id='39' "
 'data-category=\'paragraph\' style=\'font-size:16px\'>시의 "건강보험 행위 급여․비급여 목록 및 '
 '급여'),
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
 'indexing': {'chunk_id': 'chunk_000563',
              'chunk_char_len': 292,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
