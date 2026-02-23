from langchain_core.documents import Document

chunk = Document(
    page_content=('. 계약자, 피보험자 또는 보험수익자가 보험금을 지급받을 목적으로 고의로 보험금 지<br>급사유를 발생시킨 경우<br>2. 계약자, '
 '피보험자 또는 보험수익자가 보험금 청구에 관한 서류에 고의로 사실과 다<br>른 것을 기재하였거나 그 서류 또는 증거를 위조 또는 변조한 '
 '경우'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'claim', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000171',
              'chunk_char_len': 153,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
