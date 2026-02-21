from langchain_core.documents import Document

chunk = Document(
    page_content=('style=\'font-size:16px\'>시의 "건강보험 행위 급여․비급여 목록 및 급여 상대가치점수"에 따라 이미 '
 "보험</p><br><h1 id='40' style='font-size:16px'>금 지급여부가 판단된 경우에는 이를 다시 판단하지 "
 "않습니다.</h1><br><p id='41' data-category='paragraph' "
 "style='font-size:14px'>별</p><br><p id='42' data-category='paragraph' "
 "style='font-size:14px'>약</p><br><p"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000564',
              'chunk_char_len': 292,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
