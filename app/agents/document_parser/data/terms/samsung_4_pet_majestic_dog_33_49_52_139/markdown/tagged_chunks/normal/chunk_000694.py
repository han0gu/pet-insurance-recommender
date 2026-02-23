from langchain_core.documents import Document

chunk = Document(
    page_content=('- 우는 제외합니다.)\n'
 '# 제3조 (갱신계약의 보험계약 적용 특칙)제2조에 따라 갱신된 갱신계약의 경우 아래에 정한 사항을 따릅니다.# 1. 제도 및 보험료의 '
 '적용갱신계약의 약관은 갱신전 계약의 약관을 적용하고, 갱신계약의 보험요율에 관한\n'
 '제도 또는 보험료(이하 「보험요율 제도 또는 보험료」 라 합니다)는 갱신일 현재의# 보험요율 제도 또는 보험료를 적용합니다. 단, 법령 '
 '및 표준약관의 제·개정 또는\n'
 '금융위원회의 명령에 따라 약관이 개정된 경우에는 갱신일 현재의 약관을 적용합'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000694',
              'chunk_char_len': 268,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
