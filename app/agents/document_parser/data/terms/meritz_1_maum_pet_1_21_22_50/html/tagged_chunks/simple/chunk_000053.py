from langchain_core.documents import Document

chunk = Document(
    page_content=(". 기타 보험수익자가 보험금의 수령에 필요하여 제출하는 서류</p><br><p id='60' "
 "data-category='paragraph' style='font-size:14px'>② 제1항 제2호의 사고증명서는 수의사법 "
 '제12조(진단서 등)에서 규정한 내용에 따라 국<br>내의 동물병원에서 수의사에 의해 발급한 것이어야 합니다.</p><br><p '
 "id='61' data-category='paragraph' style='font-size:14px'>【수의사법 제12조(진단서 "
 "등)】</p><br><p id='62'"),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000053',
              'chunk_char_len': 289,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
