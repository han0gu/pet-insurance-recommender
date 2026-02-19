from langchain_core.documents import Document

chunk = Document(
    page_content=('<용어풀이>\n'
 '[자기공명영상(MRI)]\n'
 '강한 자기장 내에서 인체에 고주파를 전사해서 반향 되는 전자기파를 측정하는 영상진단법 [컴퓨터단층촬영(CT)] X선을 투과시켜 그 '
 '흡수차이를 컴퓨터로 재구성하여 신체의 단면영상을 얻거나 3차원적인 입체영 상을 얻는 영상진단법\n'
 '제4조 (보험금을 지급하지 않는 사유)'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 118},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000752',
              'chunk_char_len': 168,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
