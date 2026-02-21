from langchain_core.documents import Document

chunk = Document(
    page_content=('[보험료 할증]\n'
 '일반적인 경우보다 위험이 높은 피보험자가 가입하기 위한 방법의 하나로, 보험 가입 후 기간이 경\n'
 '과함에 따라 위험의 크기 및 정도가 점차 증가하는 위험 또는 기간의 경과에 상관없이 일정한 상태\n'
 '를 유지하는 위험에 적용하는 방법으로 위험 정도에 따라 특별보험료를 추가로 부가하는 방법을\n'
 '말합니다.- ④ 회사는 이 특별약관의 청약을 받고 제1회 보험료를 받은 경우에 건강진단을 받지 않\n'
 '- 는 특별약관은 청약일, 진단계약은 진단일(재진단의 경우에는 최종진단일)부터 30일'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000207',
              'chunk_char_len': 270,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
