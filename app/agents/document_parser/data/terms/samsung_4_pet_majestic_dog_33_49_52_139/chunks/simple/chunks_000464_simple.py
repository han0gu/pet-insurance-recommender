from langchain_core.documents import Document

chunk = Document(
    page_content=('<관련법규>\n'
 '[의료법 제3조(의료기관)에 규정한 종합병원] 100개 이상의 병상 구비, 병상수에 따라 일정 개수의 진료과목을 갖추고, 각 진료과목마다 '
 '전속하 는 전문의를 둔 병원을 말합니다.\n'
 '제 4조 (보험금을 지급하지 않는 사유)\n'
 '회사는 아래의 사유를 원인으로 하여 생긴 손해는 보상하지 않습니다.'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 87},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000464',
              'chunk_char_len': 167,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
