from langchain_core.documents import Document

chunk = Document(
    page_content=('- (통신판매계약의 경우 통신수단) 등을 통해 확인하고, 자동갱신 의사가 확인되는 경\n'
 '- 우 갱신전 계약은 갱신일에 갱신일 현재의 약관 등으로 갱신됩니다. 다만, 계약자가\n'
 '- 자동갱신을 원하지 않는 경우에는 갱신일에 갱신전 계약은 만료됩니다.\n'
 '- 3. 회사가 계약자의 자동갱신 의사를 확인하지 못한 경우(계약자와 연락두절 등으로 회\n'
 '- 사 안내가 계약자에게 도달하지 못한 경우 포함)에는 갱신일에 갱신일 현재의 약관\n'
 '- 등으로 갱신됩니다. 다만, 계약자는 갱신일 현재의 약관 등에 대해 갱신일로부터 90'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000665',
              'chunk_char_len': 281,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
