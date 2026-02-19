from langchain_core.documents import Document

chunk = Document(
    page_content=('개인형 이동장치를 포함하며, 장애인 또는 교통약자가 사용하는 보행보조용 의자차인 전동휠체어, 의 료용 스쿠터 등은 제외됩니다.\n'
 '※유의사항 관련 예시\n'
 'A씨(피보험자)는 일반 사무직으로 근무하던 중 상해보험을 가입하고 몇 년 후 물품배달원으로 직업을 변경하였으나 이를 고의 또는 중대한 '
 '과실로 보험회사에 알리지 않았고, 물품 배달 업무 중 일반상해로 사고가 발생한 후 보험금을 청구하였으나 보험금이 약정한 보험금보다 적게 '
 '지급되었 습니다.\n'
 '제17조 (알릴 의무 위반의 효과)'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 57},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000226',
              'chunk_char_len': 263,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
