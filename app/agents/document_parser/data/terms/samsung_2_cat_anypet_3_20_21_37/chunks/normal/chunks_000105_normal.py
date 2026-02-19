from langchain_core.documents import Document

chunk = Document(
    page_content=('① 회사는 계약과 관련하여 임직원, 보험 설계사 및 대리점의 책임있는 사유로 인하여 계약자 및 피보 험자에게 발생된 손해에 대하여 관계 '
 '법령 등에 따라 손해배상의 책임을 집니다. ② 회사는 보험금 지급 거절 및 지연지급의 사유가 없음을 알았거나 알 수 있었음에도 불구하고 '
 '소를 제 기하여 계약자 또는 피보험자에게 손해를 가한 경우에는 그에 따른 손해를 배상할 책임을 집니다'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 35,
         'page': 19},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000105',
              'chunk_char_len': 208,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
