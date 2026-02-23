from langchain_core.documents import Document

chunk = Document(
    page_content=('약관에 규정하는 "골절철심제거술"은 “건강보험 행위 급여․비급여 목록 및 급여 상대가치점수" 제2부 (행위 급여 목록․상대가치점수 및 '
 '산정지침)의 제9장 (처치 및 수술 대가치점수" 제2부 (행위 급여 목록․상대가치점수 및 산정지침)의 제9장(처치 및 수술| 료) 중 '
 '다음의수가코드에 해당하는 의료행위를 | 말합니다. |\n'
 '| --- | --- |\n'
 '| 대상이 되는 항목 | 수가코드 |'),
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
 'indexing': {'chunk_id': 'chunk_000974',
              'chunk_char_len': 214,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
