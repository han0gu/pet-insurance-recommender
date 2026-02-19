from langchain_core.documents import Document

chunk = Document(
    page_content=('【부활(효력회복)】\n'
 '보험료 납입을 연체하여 계약이 해지되고 계약자가 해약환급금을 받지 않은 경우 회사가 정하는 소정의 절차에 따라 해지된 계약을 다시 '
 '되살리는 것을 말합니다.\n'
 '제29조(강제집행 등의 절차에 따라 해지된 계약의 특별부활(효력회복))'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 17},
 'term_type': 'basic',
 'clause': {'clause_type': 'definition', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000108',
              'chunk_char_len': 137,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
