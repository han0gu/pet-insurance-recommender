from langchain_core.documents import Document

chunk = Document(
    page_content=('- 이상 효력이 없습니다.\n'
 '# 제2조 (특별약관의 내용)이 특별약관은 피보험자의 위험도가 높아 계약이 불가능한 경우 이 특별약관이 정하는\n'
 '바에 따라 가입할 수 있도록 하여 보험계약의 보험기간 중 위험에 대한 보장을 받을 수\n'
 '있는 것을 주된 내용으로 합니다.# 제 3조 (특별약관의 부가조건)① 이 특별약관에 의하여 부가하는 계약조건은 피보험자의 건강상태, '
 '위험의 종류 및 정\n'
 '도에 따라 다음 중 한가지의 방법으로 부가합니다.# 1. 할증보험료법할증위험률에 의한 보험료와 표준체 보험료와의 차액을 특별약관보험료라 '
 '하며 보'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000711',
              'chunk_char_len': 290,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
