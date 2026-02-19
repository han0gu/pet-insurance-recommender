from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 회사는 계약과 관련하여 임직원, 보험설계사 및 대리점 의 책임있는 사유로 인하여 계약자, 피보험자 및 보험수익 자에게 발생된 '
 '손해에 대하여 관계 법령 등에 따라 손해배 상의 책임을 집니다. \uf000 회사는 보험금 지급 거절 및 지연지급의 사유가 없음을 '
 '알았거나 알 수 있었는데도 소를 제기하여 계약자, 피보험 자 또는 보험수익자에게 손해를 가한 경우에는 그에 따른 손해를 배상할 책임을 '
 '집니다'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 85},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000172',
              'chunk_char_len': 218,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
