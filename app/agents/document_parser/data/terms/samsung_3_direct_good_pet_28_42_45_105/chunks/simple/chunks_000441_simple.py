from langchain_core.documents import Document

chunk = Document(
    page_content=('① 회사는 계약과 관련하여 임직원, 보험설계사 및 대리점의 책임있는 사유로 계약자 및 피보험자에게 발생된 손해에 대하여 관계 법령 등에 '
 '따라 손해배상의 책임을 집니다. ② 회사는 보험금 지급 거절 및 지연지급의 사유가 없음을 알았거나 알 수 있었는데도 소 를 제기하여 '
 '계약자 또는 피보험자에게 손해를 가한 경우에는 그에 따른 손해를 배상 할 책임을 집니다. ③ 회사가 보험금 지급여부 및 지급금액에 관하여 '
 '현저하게 공정을 잃은 합의로 계약자 또는 피보험자에게 손해를 가한 경우에도 회사는 제2항에 따라 손해를 배상할 책임을 집니다.'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 76},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000441',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
