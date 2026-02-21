from langchain_core.documents import Document

chunk = Document(
    page_content=('바에 따라 회사가 기록 및 유지･관리하는 자료의 열람(사본\n'
 '의 제공 또는 청취를 포함한다)을 요구할 수 있습니다.\uf000 회사는 일반금융소비자인 계약자가 조정을 통하여 주장\n'
 '하는 권리나 이익의 가액이 ｢금융소비자 보호에 관한 법률｣\n'
 '제42조에서 정하는 일정 금액 이내인 분쟁사건에 대하여 조\n'
 '정절차가 개시된 경우에는 관계 법령이 정하는 경우를 제외\n'
 '하고는 소를 제기하지 않습니다.# 제40조(관할법원)이 계약에 관한 소송 및 민사조정은 계약자의 주소지를 관\n'
 '할하는 법원으로 합니다. 다만, 회사와 계약자가 합의하여'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000129',
              'chunk_char_len': 284,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
