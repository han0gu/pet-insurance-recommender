from langchain_core.documents import Document

chunk = Document(
    page_content=('[심신박약자(心神薄弱者)]\n'
 '심신상실의 상태까지는 이르지 않았으나, 마음이나 정신의 장애로 인하여 사물을 변별할 능력이나 의사를 결정할 능력이 미약한 사람을 '
 '말합니다.\n'
 '3. 계약을 체결할 때 계약에서 정한 피보험자의 나이에 미달되었거나 초과되었을 경 우. 다만, 회사가 나이의 착오를 발견하였을 때 이미 '
 '계약나이에 도달한 경우에는 유효한 계약으로 보나, 제2호의 만 15세 미만자에 관한 예외가 인정되는 것은 아 닙니다.\n'
 '② 제1항에서 정하지 않은 사항은 기본계약의 계약의 무효 사항을 준용합니다.\n'
 '제23조 (특별약관 내용의 변경 등)'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 60},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000264',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
