from langchain_core.documents import Document

chunk = Document(
    page_content=('제4조 (보험금 지급에 관한 세부규정)\n'
 '보험수익자와 회사가 제3조(보험금의 지급사유)의 보험금 지급사유에 대해 합의하지 못 할 때는 보험수익자와 회사가 함께 제3자를 정하고 그 '
 '제3자의 의견에 따를 수 있습니다. 제3자는 동물병원 소속 수의사 중에 정하며, 보험금 지급사유 판정에 드는 의료비용은 회 사가 전액 '
 '부담합니다.\n'
 '제5조 (피보험자의 범위)\n'
 '이 특별약관에서 피보험자라 함은 아래에 정한 보험증권에 기재된 피보험자 및 그 가족 을 말합니다.'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 100},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000545',
              'chunk_char_len': 250,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
