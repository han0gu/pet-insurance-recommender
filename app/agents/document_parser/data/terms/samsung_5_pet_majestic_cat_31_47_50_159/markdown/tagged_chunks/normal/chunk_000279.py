from langchain_core.documents import Document

chunk = Document(
    page_content=('를 포함한다)을 요구할 수 있습니다.② 회사는 일반금융소비자인 계약자가 조정을 통하여 주장하는 권리나 이익의 가액이 ｢\n'
 '금융소비자 보호에 관한 법률｣ 제42조에서 정하는 일정 금액 이내인 분쟁사건에 대하\n'
 '여 조정절차가 개시된 경우에는 관계 법령이 정하는 경우를 제외하고는 소를 제기하\n'
 '지 않습니다.# 제 39조 (관할법원)이 특별약관에 관한 소송 및 민사조정은 계약자의 주소지를 관할하는 법원으로 합니다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000279',
              'chunk_char_len': 226,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
