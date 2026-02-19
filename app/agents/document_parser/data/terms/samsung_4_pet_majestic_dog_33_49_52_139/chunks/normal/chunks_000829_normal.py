from langchain_core.documents import Document

chunk = Document(
    page_content=('<용어풀이>\n'
 '개인형 이동장치(전동킥보드, 전동이륜평행차, 전동기의 동력만으로 움직일 수 있는 자전거 등을 포함하며, 장애인 또는 교통약자가 사용하는 '
 "보행 보조용 의자차인 전동휠체어, 의료용 스쿠터 등 은 제외합니다)는 자동차관리법에 정한 '이륜자동차', 도로교통법에 정한 "
 "'원동기장치자전거'에 포 함됩니다.\n"
 '③ 제2항에서 "그와 유사한 구조로 되어 있는 자동차"는 다음 각 호에 해당하는 자동차 를 포함합니다.\n'
 '1. 이륜인 자동차에 측차를 붙인 자동차 2. 조향장치의 조작방식, 동력전달방식 또는 원동기 냉각방식 등이 이륜의 자동차와'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 132},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000829',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
