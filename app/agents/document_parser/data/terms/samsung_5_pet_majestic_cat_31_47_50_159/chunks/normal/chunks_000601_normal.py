from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 보험가입금액(배상책임의 경우 보상한도액)을 감액할 때 해약환급 금이 없거나 최초 가입할 때 안내한 해약환급금보다 적어질 수 '
 '있습니다. ④ 회사는 제1항 제4호에 따라 계약자를 변경한 경우, 변경된 계약자에게 보험증권 및 약 관을 드리고, 변경된 계약자가 '
 '요청하는 경우 약관의 중요한 내용을 설명하여 드립니 다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 103},
 'term_type': 'special',
 'clause': {'clause_type': 'limit', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000601',
              'chunk_char_len': 179,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
