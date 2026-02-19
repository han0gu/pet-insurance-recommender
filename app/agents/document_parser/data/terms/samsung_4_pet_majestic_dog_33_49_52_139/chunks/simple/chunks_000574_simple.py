from langchain_core.documents import Document

chunk = Document(
    page_content=('1. 소송제기 2. 분쟁조정 신청 3. 수사기관의 조사 4. 해외에서 발생한 보험사고에 대한 조사 5. 제6항에 따른 회사의 조사요청에 '
 '대한 동의 거부 등 계약자, 피보험자 또는 보험수 익자의 책임있는 사유로 보험금 지급사유의 조사 및 확인이 지연되는 경우 6. '
 '제4조(보험금 지급에 관한 세부규정)에 따라 보험금 지급사유에 대해 제3자의 의견 에 따르기로 한 경우\n'
 '③ 제2항에 의하여 추가적인 조사가 이루어지는 경우, 회사는 보험수익자의 청구에 따라 회사가 추정하는 보험금의 50% 상당액을 '
 '가지급보험금으로 지급합니다.\n'
 '<용어풀이>'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 102},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000574',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
