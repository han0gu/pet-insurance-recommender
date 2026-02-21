from langchain_core.documents import Document

chunk = Document(
    page_content=('게 즉시 통지합니다. 다만, 지급예정일은 다음 각 호의 어느 하나에 해당하는 경우를\n'
 '제외하고는 제8조(보험금의 청구)에서 정한 서류를 접수한 날부터 30영업일 이내에서\n'
 '정합니다.1. 소송제기\n'
 '2. 분쟁조정 신청\n'
 '3. 수사기관의 조사\n'
 '4. 제5항에 따른 회사의 조사요청에 대한 동의 거부 등 계약자, 피보험자 또는 보험수\n'
 '익자의 책임있는 사유로 보험금 지급사유의 조사와 확인이 지연되는 경우'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000035',
              'chunk_char_len': 217,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
