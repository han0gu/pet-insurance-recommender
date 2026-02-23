from langchain_core.documents import Document

chunk = Document(
    page_content=('- ④ 제10조(환급금의 중도인출) 제1항에 따라 환급금을 중도인출한 경우에는 중도인출금\n'
 '- 및 중도인출금에 부리되었을 이자만큼 만기환급금에서 차감하여 계산하므로 제1항에\n'
 '- 정한 지급금이 감소합니다.\n'
 '- ⑤ 제24조(계약내용의 변경 등) 제1항 제5호에서 정한 적립보험료 등을 감액할 경우 제1\n'
 '- 항에 정한 만기환급금은 없거나 최초가입시 안내한 금액보다 적어질 수 있습니다.\n'
 '# 제12조 (보험금 받는 방법의 변경)① 계약자(보험금 지급사유 발생 후에는 보험수익자)는 회사의 사업방법서에서 정한 바에'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000042',
              'chunk_char_len': 281,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
