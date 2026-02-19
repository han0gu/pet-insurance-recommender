from langchain_core.documents import Document

chunk = Document(
    page_content=('이 계약에 있어서 「입원」이라 함은 수의사가 상해 또는 질병의 치료가 필요하다고 인정한 경 우로서, 자택 등에서의 치료가 곤란하여 '
 '동물병원에 입실하여 수의사의 관리 하에 치료에 전념 하는 것을 말합니다.\n'
 '제7조(보험금 지급사유의 통지)\n'
 '계약자 또는 피보험자나 보험수익자는 제4조(보험금의 지급사유)에서 정한 보험금 지급사 유의 발생을 안 때에는 지체없이 그 사실을 회사에 '
 '알려야 합니다.\n'
 '제8조(보험금의 청구)\n'
 '① 보험수익자는 다음의 서류를 제출하고 보험금을 청구하여야 합니다.'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 6},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000032',
              'chunk_char_len': 267,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
