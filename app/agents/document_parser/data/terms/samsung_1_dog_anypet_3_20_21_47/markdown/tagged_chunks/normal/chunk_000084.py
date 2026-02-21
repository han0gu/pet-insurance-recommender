from langchain_core.documents import Document

chunk = Document(
    page_content=('- 조정을 신청할 수 있으며, 분쟁조정 과정에서 계약자는 관계 법령이 정하는 바에 따라 회사가 기록\n'
 '- 및 유지·관리하는 자료의 열람(사본의 제공 또는 청취를 포함한다)을 요구할 수 있습니다.\n'
 '- ② 회사는 일반금융소비자인 계약자가 조정을 통하여 주장하는 권리나 이익의 가액이 「금융소비자보\n'
 '- 호에 관한 법률」 제42조에서 정하는 일정 금액 이내인 분쟁사건에 대하여 관계 법령이 정하는 경\n'
 '- 우를 제외하고는 소를 제기하지 않습니다.'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000084',
              'chunk_char_len': 243,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
