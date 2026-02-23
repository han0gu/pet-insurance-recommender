from langchain_core.documents import Document

chunk = Document(
    page_content=('익자의 책임있는 사유로 보험금 지급사유의 조사와 확인이 지연되는 경우\n'
 '5. 제7항에 따라 보험금 지급사유에 대해 제3자의 의견에 따르기로 한 경우③ 제2항에 의하여 추가적인 조사가 이루어지는 경우, 회사는 '
 '보험수익자의 청구에 따라\n'
 '회사가 추정하는 보험금의 50% 상당액을 가지급보험금으로 지급합니다.【가지급보험금】보험금이 지급기한 내에 지급되지 못할 것으로 판단되는 '
 '경우 회사가 예상되는 보험\n'
 '금의 일부를 먼저 지급하는 제도로 피보험자가 필요로 하는 비용을 보전해 주기 위'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage',
            'risk_domains': ['digestive', 'urinary']},
 'indexing': {'chunk_id': 'chunk_000036',
              'chunk_char_len': 264,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
