from langchain_core.documents import Document

chunk = Document(
    page_content=('- ④ 회사는 계약자가 제1항 제5호에 따라 보험가입금액을 감액하고자 할 때에는 그 감액\n'
 '- 된 부분은 해지된 것으로 보며, 이로써 회사가 지급하여야 할 해약환급금이 있을 때에\n'
 '- 는 제36조(해약환급금) 제1항에 따른 해약환급금을 계약자에게 지급합니다. 다만, 보\n'
 '- 험가입금액을 감액할 때 해약환급금이 없거나 최초 가입할 때 안내한 해약환급금보다\n'
 '- 적어질 수 있습니다.\n'
 '- ⑤ 계약자가 제2항에 따라 보험수익자를 변경하고자 할 경우 계약자와 피보험자가 동일'),
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
 'indexing': {'chunk_id': 'chunk_000097',
              'chunk_char_len': 258,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
