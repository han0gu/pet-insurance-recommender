from langchain_core.documents import Document

chunk = Document(
    page_content=('가 알리지 않은 경우 회사가 알고 있는 최종의 주소 또는\n'
 '연락처로 등기우편 등 우편물에 대한 기록이 남는 방법으로\n'
 '회사가 알린 사항은 일반적으로 도달에 필요한 기간이 지난\n'
 '때에는 계약자 또는 피보험자에게 도달한 것으로 봅니다.# 제13조(알릴 의무 위반의 효과)\uf000 회사는 아래와 같은 사실이 있을 '
 '경우에는 손해의 발생181여부에 관계없이 그 사실을 안 날부터 1개월 이내에 이 계\n'
 '약을 해지할 수 있습니다.- ① 계약자, 피보험자 또는 이들의 대리인이 고의 또는 중\n'
 '- 대한 과실로 반려동물 비용손해 관련 특별약관 일반조'),
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
 'indexing': {'chunk_id': 'chunk_000507',
              'chunk_char_len': 291,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
