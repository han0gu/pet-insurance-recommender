from langchain_core.documents import Document

chunk = Document(
    page_content=('제34조 (회사의 파산선고와 해지)\n'
 '① 회사가 파산의 선고를 받은 때에는 계약자는 계약을 해지할 수 있습니다. ② 제1항의 규정에 따라 해지하지 않은 계약은 파산선고 후 '
 '3개월이 지난 때에는 그 효 력을 잃습니다. ③ 제1항의 규정에 따라 계약이 해지되거나 제2항의 규정에 따라 계약이 효력을 잃는 경 우에 '
 '회사는 제35조(해약환급금) 제1항에 의한 해약환급금을 계약자에게 지급합니다.\n'
 '제 35조 (해약환급금)'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 62},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000309',
              'chunk_char_len': 229,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
