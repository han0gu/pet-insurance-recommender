from langchain_core.documents import Document

chunk = Document(
    page_content=('급여․비급여 목록 및 급여 상대가치점수"에 따라 이<br>미 보험금 지급여부가 판단된 경우에는 이를 다시 판단하지 '
 '않습니다.<br>\uf000 제1항의 "골절철심제거술"는 국민건강보험법에서 정한 요양급여 또는 의료급여법</p><br><p '
 "id='2' data-category='paragraph' style='font-size:14px'>에서 정한 의료급여의 절차를 거쳐 "
 "급여항목이 발생한 경우에 한합니다.</p><p id='3' data-category='paragraph' "
 "style='font-size:14px'>제5조(보험금을 지급하지"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000539',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
