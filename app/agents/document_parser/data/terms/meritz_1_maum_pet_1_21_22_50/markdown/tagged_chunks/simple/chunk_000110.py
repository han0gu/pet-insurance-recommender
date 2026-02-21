from langchain_core.documents import Document

chunk = Document(
    page_content=('- 원장에게 조정을 신청할 수 있으며, 분쟁조정 과정에서 계약자는 관계 법령이 정하는\n'
 '- 바에 따라 회사가 기록 및 유지･관리하는 자료의 열람(사본의 제공 또는 청취를 포함한\n'
 '- 다)을 요구할 수 있습니다.\n'
 '- ② 회사는 일반금융소비자인 계약자가 조정을 통하여 주장하는 권리나 이익의 가액이 ｢금\n'
 '- 융소비자보호에 관한 법률｣ 제42조에서 정하는 일정 금액 이내인 분쟁사건에 대하여\n'
 '- 조정절차가 개시된 경우에는 관계 법령이 정하는 경우를 제외하고는 소를 제기하지 않\n'
 '- 습니다.'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000110',
              'chunk_char_len': 268,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
