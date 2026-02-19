from langchain_core.documents import Document

chunk = Document(
    page_content=('. ② 제1항의 「수탁기관」 이라 함은 동물보호법 시행규칙 제43조(등록영업의 세부 범위) 에서 정하는 동물위탁관리업자로써, 반려동물 '
 '소유자의 위탁을 받아 반려동물을 영업 장 내에서 일시적으로 사육, 훈련 또는 보호하는 영업을 행하는 시설을 말합니다. ③ 제1항의 반려견 '
 '위탁비용은 위탁1일당 이 특별약관의 보험가입금액을 한도로 합니다. ④ 제1항의 경우 피보험자가 동일한 질병의 치료를 직접 목적으로 2회 '
 '이상 입원한 경우 이를 1회 입원으로 보아 입원일수를 더합니다'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 93},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000600',
              'chunk_char_len': 262,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
