from langchain_core.documents import Document

chunk = Document(
    page_content=('② 회사가 제1항에 따라 계약을 해지한 경우 회사는 그 취지를 계약자에게 통지하고 제 36조(해약환급금) 제1항에 따른 해약환급금을 '
 '지급합니다. 다만, 제1항 제1호에서 보 험수익자가 보험금의 일부 보험수익자인 경우에는 지급하지 않은 보험금에 해당하는 해약환급금을 '
 '계약자에게 지급합니다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 44},
 'term_type': 'basic',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000153',
              'chunk_char_len': 160,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
