from langchain_core.documents import Document

chunk = Document(
    page_content=('한 경우에는 서면(등기우편 등)으로 다시 알려드립니다.\n'
 '⑤ 제1항 제2호에 의한 계약의 해지가 보험금 지급사유 발생 후에 이루어진 경우에는 제\n'
 '16조(계약 후 알릴 의무) 제4항 또는 제5항에 따라 보험금을 지급합니다.\n'
 '⑥ 제1항에도 불구하고 알릴 의무를 위반한 사실이 보험금 지급사유 발생에 영향을 미쳤\n'
 '음을 회사가 증명하지 못한 경우에는 제4항 및 제5항에 관계없이 약정한 보험금을 지\n'
 '급합니다.\n'
 '⑦ 회사는 다른 보험가입내역에 대한 계약 전 알릴 의무 위반을 이유로 계약을 해지하거나\n'
 '보험금 지급을 거절하지 않습니다.'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000060',
              'chunk_char_len': 291,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
