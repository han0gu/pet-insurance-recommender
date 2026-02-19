from langchain_core.documents import Document

chunk = Document(
    page_content=('① 회사는 계약과 관련하여 임직원, 보험설계사 및 대리점의 책임있는 사유로 계약자, 피 보험자 및 보험수익자에게 발생된 손해에 대하여 '
 '관계 법령 등에 따라 손해배상의 책 임을 집니다. ② 회사는 보험금 지급 거절 및 지연지급의 사유가 없음을 알았거나 알 수 있었는데도 소 '
 '를 제기하여 계약자, 피보험자 또는 보험수익자에게 손해를 가한 경우에는 그에 따른 손해를 배상할 책임을 집니다'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 68},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000335',
              'chunk_char_len': 213,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
