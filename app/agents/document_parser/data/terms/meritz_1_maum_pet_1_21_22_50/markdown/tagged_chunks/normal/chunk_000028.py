from langchain_core.documents import Document

chunk = Document(
    page_content=('우로서, 자택 등에서의 치료가 곤란하여 동물병원에 입실하여 수의사의 관리 하에 치료에 전념\n'
 '하는 것을 말합니다.# 제7조(보험금 지급사유의 통지)계약자 또는 피보험자나 보험수익자는 제4조(보험금의 지급사유)에서 정한 보험금 '
 '지급사\n'
 '유의 발생을 안 때에는 지체없이 그 사실을 회사에 알려야 합니다.# 제8조(보험금의 청구)① 보험수익자는 다음의 서류를 제출하고 보험금을 '
 '청구하여야 합니다.- 1. 청구서(회사양식)\n'
 '- 2. 사고증명서(동물병원 진료비 영수증, 동물병원 진료비세부내역서(진료 항목별 영수금'),
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
 'indexing': {'chunk_id': 'chunk_000028',
              'chunk_char_len': 280,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
