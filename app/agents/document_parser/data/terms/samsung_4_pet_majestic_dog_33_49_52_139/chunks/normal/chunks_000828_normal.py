from langchain_core.documents import Document

chunk = Document(
    page_content=(". ② 제1항의 '이륜자동차'라 함은 자동차관리법(하위 법령, 규칙 포함)에 정한 이륜자동차 로 총배기량 또는 정격출력의 크기와 관계없이 "
 '1인 또는 2인의 사람을 운송하기에 적 합하게 제작된 이륜의 자동차 및 그와 유사한 구조로 되어 있는 자동차를 말하며, 도 로교통법(하위 '
 "법령, 규칙 포함)에 정한 '원동기장치자전거'를 포함합니다."),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 132},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000828',
              'chunk_char_len': 187,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
