from langchain_core.documents import Document

chunk = Document(
    page_content=('보험금 지급이 제한될 수 있습니다.\n'
 '원동기장치 자전거는 전동킥보드, 전동이륜평행차, 전동기의 동력만으로 움직일 수 있는 자전거 등\n'
 '개인형 이동장치를 포함하며,\n'
 '장애인 또는 교통약자가 사용하는 보행보조용 의자차인 전동휠체어, 의료용 스쿠터 등은 제외됩니다.# ※유의사항 관련 예시A씨(피보험자)는 '
 '일반 사무직으로 근무하던 중 상해보험을 가입하고 몇 년 후 물품배달원으로\n'
 '직업을 변경하였으나 이를 고의 또는 중대한 과실로 보험회사에 알리지 않았고, 물품 배달 업무 중'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000044',
              'chunk_char_len': 259,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
