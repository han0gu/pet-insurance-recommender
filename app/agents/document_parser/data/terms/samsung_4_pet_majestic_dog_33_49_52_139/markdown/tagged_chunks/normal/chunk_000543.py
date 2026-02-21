from langchain_core.documents import Document

chunk = Document(
    page_content=('약자와의 연락두절로 회사의 안내가 계약자에게 도달하지 못한 경우 포함)에는 직전\n'
 '계약과 동일한 조건으로 보험계약을 연장합니다. 다만, 보험계약이 연장된 경우 연장\n'
 '된 날 기준으로 매년 현재의 예정기초율(적용이율, 적용위험률, 부가보험요율) 적용\n'
 '및 반려동물의 연령 증가 등의 사유로 보험요율이 변동될 수 있으며 이 때의 보험료\n'
 '는「보험료 및 해약환급금 산출방법서」에 따라 산출합니다. 또한, 보험계약의 연장은\n'
 '기본계약의 보험기간 내에서만 가능합니다.'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000543',
              'chunk_char_len': 251,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
