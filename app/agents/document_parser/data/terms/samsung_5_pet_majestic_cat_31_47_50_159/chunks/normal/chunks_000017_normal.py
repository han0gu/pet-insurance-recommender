from langchain_core.documents import Document

chunk = Document(
    page_content='. 다만, 장해분류표의 각 신체부위별 판정기준에서 별도로 정한 경 우에는 그 기준에 따릅니다',
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 32},
 'term_type': 'basic',
 'clause': {'clause_type': 'definition',
            'risk_domains': ['head', 'joint', 'other']},
 'indexing': {'chunk_id': 'chunk_000017',
              'chunk_char_len': 51,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.9}},
)
