from langchain_core.documents import Document

chunk = Document(
    page_content=('보험금 지급이 제한될 수 있습니다.\n'
 '원동기장치 자전거는 전동킥보드, 전동이륜평행차, 전동기의 동력만으로 움직일 수 있는 자전거 등- 56 -개인형 이동장치를 포함하며, '
 '장애인 또는 교통약자가 사용하는 보행보조용 의자차인 전동휠체어, 의\n'
 '료용 스쿠터 등은 제외됩니다.# ※유의사항 관련 예시A씨(피보험자)는 일반 사무직으로 근무하던 중 상해보험을 가입하고 몇 년 후 '
 '물품배달원으로\n'
 '직업을 변경하였으나 이를 고의 또는 중대한 과실로 보험회사에 알리지 않았고, 물품 배달 업무 중'),
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
 'indexing': {'chunk_id': 'chunk_000195',
              'chunk_char_len': 265,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
