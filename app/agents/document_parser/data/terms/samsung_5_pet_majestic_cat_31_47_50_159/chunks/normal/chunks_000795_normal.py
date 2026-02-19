from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 계약자는 갱신일 현재의 약관 등에 대해 갱신일로부터 90 일 이내에 그 계약을 취소할 수 있으며, 이 경우 회사는 갱신일 이후 '
 '계약자가 납입 한 해당 갱신계약의 보험료를 돌려 드립니다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 124},
 'term_type': 'special',
 'clause': {'clause_type': 'renewal', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000795',
              'chunk_char_len': 109,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
