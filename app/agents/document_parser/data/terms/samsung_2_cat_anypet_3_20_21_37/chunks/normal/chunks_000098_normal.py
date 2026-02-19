from langchain_core.documents import Document

chunk = Document(
    page_content=('제31조(분쟁의 조정)\n'
 '① 계약에 관하여 분쟁이 있는 경우 분쟁 당사자 또는 기타 이해관계인과 회사는 금융감독원장에게 조정을 신청할 수 있으며, 분쟁조정 '
 '과정에서 계약자는 관계 법령이 정하는 바에 따라 회사가 기록 및 유지·관리하는 자료의 열람(사본의 제공 또는 청취를 포함한다)을 요구할 '
 '수 있습니다. ② 회사는 일반금융소비자인 계약자가 조정을 통하여 주장하는 권리나 이익의 가액이 「금융소비자보 호에 관한 법률」 '
 '제42조에서 정하는 일정 금액 이내인 분쟁사건에 대하여 관계 법령이 정하는 경 우를 제외하고는 소를 제기하지 않습니다.'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 35,
         'page': 18},
 'term_type': 'basic',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000098',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
