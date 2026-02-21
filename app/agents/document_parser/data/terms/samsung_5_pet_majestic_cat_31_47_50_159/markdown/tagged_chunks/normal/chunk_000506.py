from langchain_core.documents import Document

chunk = Document(
    page_content=('- 감액하고자 할 때에는 그 감액된 부분은 특별약관이 해지된 것으로 보며, 이로써 회사\n'
 '- 가 지급하여야 할 해약환급금이 있을 때에는 이 특별약관의 해약환급금을 계약자에게\n'
 '- 지급합니다. 다만, 보험가입금액(배상책임의 경우 보상한도액)을 감액할 때 해약환급\n'
 '- 금이 없거나 최초 가입할 때 안내한 해약환급금보다 적어질 수 있습니다.\n'
 '- ④ 회사는 제1항 제4호에 따라 계약자를 변경한 경우, 변경된 계약자에게 보험증권 및 약\n'
 '- 관을 드리고, 변경된 계약자가 요청하는 경우 약관의 중요한 내용을 설명하여 드립니\n'
 '- 다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'limit', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000506',
              'chunk_char_len': 290,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
