from langchain_core.documents import Document

chunk = Document(
    page_content=('전치 2개월 초과 4개월 이하 | 150만원\n'
 '전치 1개월 초과 2개월 이하 | 100만원\n'
 '② 제1항에서 「상해, 폭행 및 폭력」의 예상치료기간은 관할 검·경찰 기관에 피해 입 증을 위해 제출한 서류(진단서 또는 상해진단서, '
 '법원의 판결문 등)에 기재된 향후 치 료의견을 기초로 합니다. ③ 하나의 사건이 다수의 강력범죄에 해당하는 경우에는 그 중 지급금액이 '
 '가장 큰 하나 의 강력범죄에 대해서만 보험금을 지급합니다.\n'
 '제2조 (강력범죄의 정의)\n'
 '① 이 특별약관에서 강력범죄는 아래의 항목에 해당하는 죄를 말합니다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 85},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000457',
              'chunk_char_len': 286,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
