from langchain_core.documents import Document

chunk = Document(
    page_content=('- 제35조(해약환급금) 제1항에 따른 해약환급금을 계약자에게 지급합니다.\n'
 '- ② 제22조(특별약관의 무효) 에 따라 사망을 보험금 지급사유로 하는 계약에서 서면으로\n'
 '- 동의를 한 피보험자는 계약의 효력이 유지되는 기간에는 언제든지 서면동의를 장래를\n'
 '- 향하여 철회할 수 있으며, 서면동의 철회로 계약이 해지되어 회사가 지급하여야 할 해\n'
 '약환급금이 있을 때에는 제35조(해약환급금) 제1항에 따른 해약환급금을 계약자에게'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000258',
              'chunk_char_len': 234,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
