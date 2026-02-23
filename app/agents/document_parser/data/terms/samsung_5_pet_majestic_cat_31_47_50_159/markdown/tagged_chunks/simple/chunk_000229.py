from langchain_core.documents import Document

chunk = Document(
    page_content=('보험금을 지급한 경우 변경된 보험수익자에게는 별도로 보험금을 지급하지 않습니다.- ③ 회사는 계약자가 제1항에 따라 이 특별약관의 '
 '보험가입금액을 감액하고자 할 때에는\n'
 '- 그 감액된 부분은 해지된 것으로 보며, 이로써 회사가 지급하여야 할 해약환급금이 있\n'
 '- 을 때에는 제35조(해약환급금) 제1항에 따른 해약환급금을 계약자에게 지급합니다.\n'
 '- 다만, 보험가입금액을 감액할 때 해약환급금이 없거나 최초 가입할 때 안내한 해약환\n'
 '- 급금보다 적어질 수 있습니다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000229',
              'chunk_char_len': 256,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
