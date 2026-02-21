from langchain_core.documents import Document

chunk = Document(
    page_content=('. 제3조(보상하는 손해) 제2호 ‘다’목 또는 ‘라’목의 비용 : 이 비용과 제1호에 의한<br>보상액의 합계액을 보상한도액내에서 '
 "보상합니다.</p><br><p id='52' data-category='paragraph' style='font-size:14px'>② "
 "보험기간 중 발생하는 사고에 대한 회사의 보상총액은 보험증권에 기재된 총 보상한도<br>액을 한도로 합니다.</p><h1 id='53' "
 "style='font-size:14px'>제9조(의무보험과의 관계)</h1><br><p id='54'"),
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
 'indexing': {'chunk_id': 'chunk_000231',
              'chunk_char_len': 282,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
