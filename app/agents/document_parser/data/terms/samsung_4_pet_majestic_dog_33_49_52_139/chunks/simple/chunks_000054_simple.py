from langchain_core.documents import Document

chunk = Document(
    page_content=('제13조 (주소변경통지)\n'
 '① 계약자(보험수익자가 계약자와 다른 경우 보험수익자를 포함합니다)는 주소 또는 연락 처가 변경된 경우에는 지체없이 그 변경내용을 회사에 '
 '알려야 합니다. ② 제1항에서 정한 대로 계약자 또는 보험수익자가 변경내용을 알리지 않은 경우에는 계 약자 또는 보험수익자가 회사에 알린 '
 '최종의 주소 또는 연락처로 등기우편 등 우편물 에 대한 기록이 남는 방법으로 회사가 알린 사항은 일반적으로 도달에 필요한 기간이 지난 '
 '때에 계약자 또는 보험수익자에게 도달된 것으로 봅니다.\n'
 '제14조 (보험수익자의 지정)'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 37},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000054',
              'chunk_char_len': 290,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
