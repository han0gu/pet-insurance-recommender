from langchain_core.documents import Document

chunk = Document(
    page_content=('가 영업용운전자로 직업 또는 직무 변경 포함)하거나 이륜자동차 또는 원동기장치 자전거를 계속\n'
 '적으로 사용하게 된 경우에는 즉시 회사에 알려야 합니다. 그러지 않을 경우 보험사고가 발생한 경\n'
 '우에도 보험금 지급이 제한될 수 있습니다.\n'
 '원동기장치 자전거는 전동킥보드, 전동이륜평행차, 전동기의 동력만으로 움직일 수 있는 자전거 등\n'
 '개인형 이동장치를 포함하며, 장애인 또는 교통약자가 사용하는 보행보조용 의자차인 전동휠체어,'),
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
 'indexing': {'chunk_id': 'chunk_000062',
              'chunk_char_len': 234,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
