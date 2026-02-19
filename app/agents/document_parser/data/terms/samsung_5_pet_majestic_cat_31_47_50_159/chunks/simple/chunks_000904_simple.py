from langchain_core.documents import Document

chunk = Document(
    page_content=('. 단, 길이가 5mm 미만의 반흔은 합산대상에서 제 외한다. 5) 추상(추한 모습)이 얼굴과 머리 또는 목 부위에 걸쳐 있는 경우에는 '
 '머리 또는 목에 있는 흉터의 길이 또는 면적의 1/2을 얼굴의 추상(추한 모습)으로 보아 산정한다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 140},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['skin', 'head']},
 'indexing': {'chunk_id': 'chunk_000904',
              'chunk_char_len': 131,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
