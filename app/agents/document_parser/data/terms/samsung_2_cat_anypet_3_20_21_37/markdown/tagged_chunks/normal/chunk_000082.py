from langchain_core.documents import Document

chunk = Document(
    page_content=('- 화 문자메시지 또는 이에 준하는 전자적 의사표시 포함)가 계약자 또는 그의 대리인에게 도달한 날\n'
 '- 로 봅니다.\n'
 '# 제7관 분쟁의 조정 등# 제31조(분쟁의 조정)- ① 계약에 관하여 분쟁이 있는 경우 분쟁 당사자 또는 기타 이해관계인과 회사는 '
 '금융감독원장에게\n'
 '- 조정을 신청할 수 있으며, 분쟁조정 과정에서 계약자는 관계 법령이 정하는 바에 따라 회사가 기록\n'
 '- 및 유지·관리하는 자료의 열람(사본의 제공 또는 청취를 포함한다)을 요구할 수 있습니다.'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000082',
              'chunk_char_len': 255,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
