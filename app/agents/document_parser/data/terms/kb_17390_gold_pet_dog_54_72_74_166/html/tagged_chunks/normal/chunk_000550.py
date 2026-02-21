from langchain_core.documents import Document

chunk = Document(
    page_content=(". 기타 보험수익자가 보험금의 수령에 필요하여 제출하는 서류</p><br><h1 id='15' "
 "style='font-size:14px'>용 어 풀 이 건강보험심사평가원 진료수가코드(EDI)</h1><br><p id='16' "
 "data-category='paragraph' style='font-size:14px'>「건강보험 행위 급여․비급여 목록 및 급여 "
 "상대가치점수(보건복지부 고시)」</p><br><p id='17' data-category='paragraph' "
 "style='font-size:14px'>에서 정한 처치 및 수술료,"),
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
 'indexing': {'chunk_id': 'chunk_000550',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
