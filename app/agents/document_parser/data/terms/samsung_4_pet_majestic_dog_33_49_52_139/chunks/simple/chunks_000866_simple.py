from langchain_core.documents import Document

chunk = Document(
    page_content=('구분 | 특 정 신 체 부 위\n'
 '26 | 난소 및 난관\n'
 '27 | 고환[고환초막(고환집막) 포함], 부고환, 정관, 정삭 및 정낭\n'
 '28 | 갑상선\n'
 '29 | 부갑상선\n'
 '30 | 서혜부(넓적다리 부위의 위쪽 주변)(서혜 탈장, 음낭 탈장 또는 대퇴 탈장이 생긴 경우 에 한함)\n'
 '31 | 피부(두피 및 입술 포함)\n'
 '32 | 경추부(해당신경 포함)\n'
 '33 | 흉추부(해당신경 포함)\n'
 '34 | 요추부(해당신경 포함)\n'
 '35 | 천골(엉치뼈)부 및 미골(꼬리뼈)부(해당 신경 포함)\n'
 '36 | 왼쪽 어깨\n'
 '37 | 오른쪽 어깨'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 137},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['skin', 'joint', 'other']},
 'indexing': {'chunk_id': 'chunk_000866',
              'chunk_char_len': 281,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
