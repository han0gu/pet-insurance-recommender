from langchain_core.documents import Document

chunk = Document(
    page_content=('1. 소송제기 2. 분쟁조정 신청 3. 수사기관의 조사 4. 해외에서 발생한 보험사고에 대한 조사 5. 제6항에 따른 회사의 조사요청에 '
 '대한 동의 거부 등 계약자, 피보험자 또는 보험수 익자의 책임있는 사유로 보험금 지급사유의 조사 및 확인이 지연되는 경우'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 33},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000031',
              'chunk_char_len': 143,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
