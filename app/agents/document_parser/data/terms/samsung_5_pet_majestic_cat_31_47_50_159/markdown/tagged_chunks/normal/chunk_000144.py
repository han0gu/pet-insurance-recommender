from langchain_core.documents import Document

chunk = Document(
    page_content=('[기명날인]\n'
 '자기 이름을 쓰고 도장을 찍는 것을 말합니다.# 제 44조 (회사의 손해배상책임)① 회사는 계약과 관련하여 임직원, 보험설계사 및 '
 '대리점의 책임있는 사유로 계약자, 피\n'
 '보험자 및 보험수익자에게 발생된 손해에 대하여 관계 법령 등에 따라 손해배상의 책# 임을 집니다.- ② 회사는 보험금 지급 거절 및 '
 '지연지급의 사유가 없음을 알았거나 알 수 있었는데도 소\n'
 '- 를 제기하여 계약자, 피보험자 또는 보험수익자에게 손해를 가한 경우에는 그에 따른\n'
 '- 손해를 배상할 책임을 집니다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000144',
              'chunk_char_len': 272,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
