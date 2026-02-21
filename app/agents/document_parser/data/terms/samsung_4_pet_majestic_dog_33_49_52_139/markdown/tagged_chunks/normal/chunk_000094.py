from langchain_core.documents import Document

chunk = Document(
    page_content=('의사를 결정할 능력이 미약한 사람을 말합니다.3. 계약을 체결할 때 계약에서 정한 피보험자의 나이에 미달되었거나 초과되었을 경\n'
 '우. 다만, 회사가 나이의 착오를 발견하였을 때 이미 계약나이에 도달한 경우에는\n'
 '유효한 계약으로 보나, 제2호의 만 15세 미만자에 관한 예외가 인정되는 것은 아\n'
 '닙니다.제24조 (계약내용의 변경 등)- 42 -# ① 계약자는 회사의 승낙을 얻어 다음의 사항을 변경할 수 있습니다. 이 경우 승낙을 '
 '서\n'
 '면 등으로 알리거나 보험증권의 뒷면에 기재하여 드립니다.- 1. 보험종목\n'
 '- 2. 보험기간'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000094',
              'chunk_char_len': 290,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
