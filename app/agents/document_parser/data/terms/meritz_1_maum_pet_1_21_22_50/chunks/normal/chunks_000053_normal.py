from langchain_core.documents import Document

chunk = Document(
    page_content=('【계약자가 2명 이상인 경우】\n'
 '계약자가 2명 이상인 경우, 보험료 납입의무 등 보험계약에 따른 계약자의 의무를 연대로 합니다.\n'
 '【연대】\n'
 '2인 이상이 함께 책임을 지므로 각자 채무의 전부를 이행할 책임을 지되(지분만큼 분할하여 책임을 지는 것과 다름), 다만 어느 1인의 '
 '이행으로 나머지 사람들도 책임 을 면하게 되는 것을 말합니다.\n'
 '제 3 관 계약자의 계약 전 알릴 의무 등\n'
 '제15조(계약 전 알릴 의무)'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 9},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000053',
              'chunk_char_len': 226,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
