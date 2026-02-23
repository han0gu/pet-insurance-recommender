from langchain_core.documents import Document

chunk = Document(
    page_content=("추가 가산한다.</h1><p id='73' data-category='paragraph' "
 "style='font-size:14px'>별표10 창상봉합술(안면/경부 외) 대상 수가코드<br>약관에 규정하는 "
 '"창상봉합술(급여)"는 “건강보험 행위 급여․비급여 목록 및 급여 상<br>대가치점수" 제2부 (행위 급여 목록․상대가치점수 및 '
 "산정지침)의 제9장(처치 및 수술</p><br><table id='74' "
 'style=\'font-size:14px\'><thead></thead><tbody><tr><td colspan="2">료) 중 다음의'),
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
 'indexing': {'chunk_id': 'chunk_001747',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
