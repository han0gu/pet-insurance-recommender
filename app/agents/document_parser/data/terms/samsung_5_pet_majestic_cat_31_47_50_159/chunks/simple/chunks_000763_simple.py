from langchain_core.documents import Document

chunk = Document(
    page_content=('. 아래의 질병으로 인하여 발생한 손해는 보상하지 않습니다. 다만, 질병의 발생일 로부터 과거 1년 이내의 예방접종 기록이 있는 경우에는 '
 '보상합니다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 119},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000763',
              'chunk_char_len': 83,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
