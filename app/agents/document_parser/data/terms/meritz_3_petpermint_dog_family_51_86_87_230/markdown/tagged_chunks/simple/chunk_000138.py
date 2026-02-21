from langchain_core.documents import Document

chunk = Document(
    page_content=('자에게 발생된 손해에 대하여 관계 법령 등에 따라 손해배\n'
 '상의 책임을 집니다.\n'
 '\uf000 회사는 보험금 지급 거절 및 지연지급의 사유가 없음을\n'
 '알았거나 알 수 있었는데도 소를 제기하여 계약자, 피보험\n'
 '자 또는 보험수익자에게 손해를 가한 경우에는 그에 따른\n'
 '손해를 배상할 책임을 집니다.\n'
 '\uf000 회사가 보험금 지급여부 및 지급금액에 관하여 현저하게\n'
 '공정을 잃은 합의로 보험수익자에게 손해를 가한 경우에도\n'
 '회사는 제2항에 따라 손해를 배상할 책임을 집니다.# 【 현저하게 공정을 잃은 합의 】사회통념상 일반 보통인이라면 그 같은 일을 하지 '
 '않을'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000138',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
