from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 계약자 또는 피보험자가 상<br>법 제657조 제1항에 의해 보험사고의 발생을 회사에 알린 경우에는 제3조(보상하는 '
 '손<br>해) 제3항 제1호 및 제2호 ‘다’목 또는 ‘라’목의 비용에 대하여 보상한도액을 한도로 보<br>상하여 드립니다.</p><h1 '
 "id='41' style='font-size:14px'>제6조(보험금의 청구)</h1><br><h1 id='42' "
 "style='font-size:14px'>피보험자가 보험금을 청구할 때에는 다음의 서류를 회사에 제출하여야 합니다.</h1><br><p "
 "id='43'"),
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
 'indexing': {'chunk_id': 'chunk_000222',
              'chunk_char_len': 295,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
