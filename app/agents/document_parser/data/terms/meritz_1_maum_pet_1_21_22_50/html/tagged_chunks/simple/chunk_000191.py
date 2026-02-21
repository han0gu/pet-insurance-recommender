from langchain_core.documents import Document

chunk = Document(
    page_content=("계약자, 피보험자 또는 보험수익자에게 손해를 가한 경우에는 그에 따른</p><footer id='116' "
 "style='font-size:14px'>- 20 -</footer><h1 id='0' style='font-size:14px'>손해를 "
 "배상할 책임을 집니다.</h1><br><p id='1' data-category='paragraph' "
 "style='font-size:14px'>③ 회사가 보험금 지급여부 및 지급금액에 관하여 현저하게 공정을 잃은 합의로 "
 '보험수익<br>자에게 손해를 가한 경우에도 회사는 제2항에 따라 손해를'),
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
 'indexing': {'chunk_id': 'chunk_000191',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
