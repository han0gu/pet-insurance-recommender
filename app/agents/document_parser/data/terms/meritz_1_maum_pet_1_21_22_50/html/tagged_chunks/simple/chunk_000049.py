from langchain_core.documents import Document

chunk = Document(
    page_content=("수의사의 관리 하에 치료에 전념<br>하는 것을 말합니다.</p><h1 id='54' "
 "style='font-size:14px'>제7조(보험금 지급사유의 통지)</h1><br><p id='55' "
 "data-category='paragraph' style='font-size:14px'>계약자 또는 피보험자나 보험수익자는 "
 '제4조(보험금의 지급사유)에서 정한 보험금 지급사<br>유의 발생을 안 때에는 지체없이 그 사실을 회사에 알려야 합니다.</p><h1 '
 "id='56' style='font-size:14px'>제8조(보험금의"),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000049',
              'chunk_char_len': 291,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
