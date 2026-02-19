from langchain_core.documents import Document

chunk = Document(
    page_content=('1. 보험기간의 변경 2. 감액완납보험으로의 변경\n'
 '<용어풀이>\n'
 '[감액완납보험] 차회 이후의 보험료 납입을 중단하는 대신 가입금액을 감액하는 보험\n'
 '제6조 (준용규정)\n'
 '이 특별약관에 정하지 않은 사항은 보험계약을 따릅니다.'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 134},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000843',
              'chunk_char_len': 122,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
