from langchain_core.documents import Document

chunk = Document(
    page_content=('. \uf000 회사는 일반금융소비자인 계약자가 조정을 통하여 주장 하는 권리나 이익의 가액이 ｢금융소비자 보호에 관한 법률｣ '
 '제42조에서 정하는 일정 금액 이내인 분쟁사건에 대하여 조 정절차가 개시된 경우에는 관계 법령이 정하는 경우를 제외 하고는 소를 제기하지 '
 '않습니다.'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 79},
 'term_type': 'basic',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000162',
              'chunk_char_len': 148,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
