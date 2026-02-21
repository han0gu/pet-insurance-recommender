from langchain_core.documents import Document

chunk = Document(
    page_content=('- 5. 이미 납입한 보험료 : 계약자가 실제로 납입한 보험료를 말합니다.\n'
 '# ④ 기간과 날짜 관련 용어- 1. 보험기간: 계약에 따라 보장을 받는 기간을 말합니다.\n'
 '- 2. 영업일: 회사가 영업점에서 정상적으로 영업하는 날을 말하며, 토요일, ‘관공서의\n'
 '- 공휴일에 관한 규정’에 따른 공휴일과 노동절을 제외합니다.\n'
 '# ⑤ 보험료 관련 용어- 1. 보장보험료와 적립보험료를 합하여 보험료라 합니다.\n'
 '- 2. 보장보험료: 손해를 보장하는데 필요한 보험료를 말합니다.'),
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
 'indexing': {'chunk_id': 'chunk_000006',
              'chunk_char_len': 260,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
