from langchain_core.documents import Document

chunk = Document(
    page_content=('② 회사는 약관의 뜻이 명백하지 않은 경우에는 계약자에게 유리하게 해석합니다. ③ 회사는 보험금을 지급하지 않는 사유 등 계약자나 '
 '피보험자에게 불리하거나 부담을 주는 내용은 확대하여 해석하지 않습니다.\n'
 '제42조 (설명서 교부 및 보험안내자료 등의 효력)'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 63},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000331',
              'chunk_char_len': 141,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
