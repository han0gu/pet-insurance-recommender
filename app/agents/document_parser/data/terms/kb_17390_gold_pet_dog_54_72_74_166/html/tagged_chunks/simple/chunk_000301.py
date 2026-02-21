from langchain_core.documents import Document

chunk = Document(
    page_content=("id='139' data-category='paragraph' style='font-size:14px'>성이 확보된 전자적 수단을 "
 "활용한</p><br><p id='140' data-category='paragraph' "
 "style='font-size:14px'>보험수익자 의사표시의 확인방법 포함)</p><br><p id='141' "
 "data-category='paragraph' style='font-size:14px'>제41조(지정대리청구인에 의한 보험금의 "
 "청구)</p><br><h1 id='142'"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000301',
              'chunk_char_len': 279,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
