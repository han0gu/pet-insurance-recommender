from langchain_core.documents import Document

chunk = Document(
    page_content=('피보험자의 요구 또는 보험료 납입 연체로 해지된 경우 유\n'
 '사계약에서 정한 부담보 기간 종료일 이내에서 계약의 부담\n'
 '보 기간을 적용하고, 유사계약에서 정한 질병과 동일하거나\n'
 '축소된 범위로 계약의 부담보 설정 범위를 정하며, 유사계\n'
 '약이 다수인 경우 반려동물에게 가장 유리한 계약조건을 적\n'
 '용합니다. 단, 계약 청약일 현재 부담보 기간을「계약의 보\n'
 '험기간」으로 적용한 유사계약이 유지중이거나, 계약 청약\n'
 '일 전 6개월 이내에 계약자 및 피보험자의 요구 또는 보험\n'
 '료 납입 연체로 해지된 경우「계약의 보험기간」이내에서'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000547',
              'chunk_char_len': 286,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
