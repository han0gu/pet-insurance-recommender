from langchain_core.documents import Document

chunk = Document(
    page_content=('[핵연료물질]\n'
 '사용된 연료를 | 포함합니다.\n'
 '[핵연료물질에 | 의하여 오염된 물질]\n'
 '원자핵 분열 생성물을 포함합니다.\n'
 '② 회사는 피보험자가 다음에 열거한 배상책임을 부담함으로써 입은 손해를 보상하지 않 습니다.'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 121},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000753',
              'chunk_char_len': 117,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
