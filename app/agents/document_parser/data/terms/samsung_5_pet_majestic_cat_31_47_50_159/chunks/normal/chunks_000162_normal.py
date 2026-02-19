from langchain_core.documents import Document

chunk = Document(
    page_content=('. ② 회사는 일반금융소비자인 계약자가 조정을 통하여 주장하는 권리나 이익의 가액이 ｢ 금융소비자 보호에 관한 법률｣ 제42조에서 정하는 '
 '일정 금액 이내인 분쟁사건에 대하 여 조정절차가 개시된 경우에는 관계 법령이 정하는 경우를 제외하고는 소를 제기하 지 않습니다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 46},
 'term_type': 'basic',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000162',
              'chunk_char_len': 148,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
