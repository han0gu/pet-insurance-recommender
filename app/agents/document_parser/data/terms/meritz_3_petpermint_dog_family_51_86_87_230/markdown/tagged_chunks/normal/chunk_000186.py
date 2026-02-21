from langchain_core.documents import Document

chunk = Document(
    page_content=('우 미리 정해진 비율로 보험금을 감액하여 지급하는 방\n'
 '법을 말합니다.# 【 보험료 할증 】일반적인 경우보다 위험이 높은 반려동물이 가입하기 위\n'
 '한 방법의 하나로, 보험 가입 후 기간이 경과함에 따라\n'
 '위험의 크기 및 정도가 점차 증가하는 위험 또는 기간의\n'
 '경과에 상관없이 일정한 상태를 유지하는 위험에 적용하\n'
 '는 방법으로 위험 정도에 따라 특별보험료를 추가로 부\n'
 '가하는 방법을 말합니다.\uf000 회사는 이 특별약관의 청약을 받고, 제1회 보험료를 받\n'
 '은 경우에 건강진단을 받지 않는 계약은 청약일, 진단계약'),
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
 'indexing': {'chunk_id': 'chunk_000186',
              'chunk_char_len': 279,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
