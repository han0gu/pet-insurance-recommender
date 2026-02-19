from langchain_core.documents import Document

chunk = Document(
    page_content=('내에서 여러 개의 척추체(척추뼈 몸통)에 압박골절이 발생한 경우에는 각 척추체(척추뼈 몸통)의 압박률을 합산하고, 두 개 이상의 '
 '운동단위에서 장 해가 발생한 경우에는 그 중 가장 높은 지급률을 적용한다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 141},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['joint', 'other']},
 'indexing': {'chunk_id': 'chunk_000912',
              'chunk_char_len': 113,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
