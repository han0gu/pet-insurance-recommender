from langchain_core.documents import Document

chunk = Document(
    page_content=('② 회사는 일반금융소비자인 계약자가 조정을 통하여 주장하는 권리나 이익의 가액이 ｢ 금융소비자 보호에 관한 법률｣ 제42조에서 정하는 '
 '일정 금액 이내인 분쟁사건에 대하 여 조정절차가 개시된 경우에는 관계 법령이 정하는 경우를 제외하고는 소를 제기하 지 않습니다.\n'
 '제 39조 (관할법원)\n'
 '이 특별약관에 관한 소송 및 민사조정은 계약자의 주소지를 관할하는 법원으로 합니다. 다만, 회사와 계약자가 합의하여 관할법원을 달리 정할 '
 '수 있습니다.\n'
 '제 40조 (소멸시효)'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 58},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000299',
              'chunk_char_len': 257,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
