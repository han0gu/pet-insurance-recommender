from langchain_core.documents import Document

chunk = Document(
    page_content=("159 -</p><br><p id='60' data-category='paragraph' style='font-size:14px'>별표9 "
 '창상봉합술(안면/경부) 대상 수가코드<br>공<br>약관에 규정하는 "창상봉합술(급여)"는 “건강보험 행위 급여․비급여 목록 및 급여 '
 "상<br>통</p><br><table id='61' "
 'style=\'font-size:16px\'><thead></thead><tbody><tr><td colspan="2">료) 중 다음의 '
 '수가코드에 해당하는 검사를'),
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
 'indexing': {'chunk_id': 'chunk_001732',
              'chunk_char_len': 269,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
