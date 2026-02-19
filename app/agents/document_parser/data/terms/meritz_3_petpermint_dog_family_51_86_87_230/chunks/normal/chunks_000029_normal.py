from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 제2항에 따라 장해지급률의 판정 및 지급할 보험금의 결 정과 관련하여 확정된 장해지급률에 따른 보험금을 초과한 부분에 대한 '
 '분쟁으로 보험금 지급이 늦어지는 경우에는 보 험수익자의 청구에 따라 이미 확정된 보험금을 먼저 가지급 합니다. \uf000 제2항에 따라 '
 '추가적인 조사가 이루어지는 경우, 회사는 보험수익자의 청구에 따라 회사가 추정하는 보험금의 50% 상당액을 가지급보험금으로 지급합니다.\n'
 '【가지급보험금】'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 57},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000029',
              'chunk_char_len': 227,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
