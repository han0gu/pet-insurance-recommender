from langchain_core.documents import Document

chunk = Document(
    page_content=('② 제1항에도 불구하고 제1항 제1호의 살인, 제4호의 「상해, 폭행 및 폭력」 등으로 피 보험자의 신체에 피해가 발생한 경우에는 '
 '1개월을 초과하여 의사의 치료를 요하는 신 체상해를 입은 때에만 보상합니다. ③ 제1항 및 제2항에도 불구하고 제1항 제2호 내지 '
 '제4호에 해당하는 강력범죄에 의하여 피보험자가 사망하였을 경우에는 제1항 제1호의 살인에 해당하는 것으로 봅니다.\n'
 '제3조 (보험금 지급에 관한 세부규정)'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 85},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000459',
              'chunk_char_len': 231,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
