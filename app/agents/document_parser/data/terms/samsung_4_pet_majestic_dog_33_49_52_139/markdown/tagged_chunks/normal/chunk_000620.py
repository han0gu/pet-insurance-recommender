from langchain_core.documents import Document

chunk = Document(
    page_content=('경우 보험계약일은 이 특별약관의 제1회 보험료를 받은 날로 합니다.- \n'
 '<예시안내># [「반려견 사망위로금」에 대한 보장개시일(책임개시일) 계산]![image](/image/placeholder)\n'
 '보험계약일 보장개시일(책임개시일)\n'
 '◄───── 30일 ─────►\n'
 '2022년 8월 1일 2022년 8월 31일# 제2조 (보험금을 지급하지 않는 사유)# 회사는 아래의 사유를 원인으로 하여 생긴 손해는 '
 '보상하지 않습니다.- 1. 계약자 및 피보험자, 이들의 가족 또는 사용인의 고의 또는 중대한 과실'),
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
 'indexing': {'chunk_id': 'chunk_000620',
              'chunk_char_len': 277,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
