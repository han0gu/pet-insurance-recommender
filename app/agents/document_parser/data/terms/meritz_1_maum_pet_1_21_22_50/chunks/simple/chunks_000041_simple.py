from langchain_core.documents import Document

chunk = Document(
    page_content=('1. 소송제기 2. 분쟁조정 신청 3. 수사기관의 조사 4. 제5항에 따른 회사의 조사요청에 대한 동의 거부 등 계약자, 피보험자 또는 '
 '보험수 익자의 책임있는 사유로 보험금 지급사유의 조사와 확인이 지연되는 경우 5. 제7항에 따라 보험금 지급사유에 대해 제3자의 의견에 '
 '따르기로 한 경우\n'
 '③ 제2항에 의하여 추가적인 조사가 이루어지는 경우, 회사는 보험수익자의 청구에 따라 회사가 추정하는 보험금의 50% 상당액을 '
 '가지급보험금으로 지급합니다.\n'
 '【가지급보험금】'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 7},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000041',
              'chunk_char_len': 257,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
