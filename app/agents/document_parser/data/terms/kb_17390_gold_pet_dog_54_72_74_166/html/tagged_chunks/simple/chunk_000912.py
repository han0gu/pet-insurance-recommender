from langchain_core.documents import Document

chunk = Document(
    page_content=('. 계약자, 피보험자 또는 보험수익자가 보험금을 지급받을 목적으로 고의로 보험<br>금 지급사유를 발생시킨 경우<br>2. 계약자, '
 '피보험자 또는 보험수익자가 보험금 청구에 관한 서류에 고의로 사실<br>과 다른 것을 기재하였거나 그 서류 또는 증거를 위조 또는 변조한 '
 '경우'),
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
 'indexing': {'chunk_id': 'chunk_000912',
              'chunk_char_len': 153,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
