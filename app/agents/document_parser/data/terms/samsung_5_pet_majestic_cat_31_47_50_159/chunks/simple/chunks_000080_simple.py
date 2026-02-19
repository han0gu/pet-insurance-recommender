from langchain_core.documents import Document

chunk = Document(
    page_content=('나 보험금 지급을 거절하지 않습니다.\n'
 '⑧ 제31조(보험료의 납입을 연체하여 해지된 계약의 부활(효력회복))에 따라 이 계약이 부활(효력회복)된 경우에는 부활(효력회복)계약을 '
 '제2항의 최초계약으로 봅니다. 부활 (효력회복)이 여러차례 발생된 경우에는 각각의 부활(효력회복)계약을 최초계약으로 봅니다.\n'
 '제19조 (사기에 의한 계약)'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 38},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000080',
              'chunk_char_len': 183,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
