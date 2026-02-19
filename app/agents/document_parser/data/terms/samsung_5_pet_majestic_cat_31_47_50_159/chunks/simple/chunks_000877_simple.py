from langchain_core.documents import Document

chunk = Document(
    page_content=('. 이 경우 안구가 적출되어 눈자위의 조직요몰(凹沒) 등으로 의 안마저 끼워 넣을 수 없는 상태이면 "뚜렷한 추상(추한 모습)" 으로, '
 '의안을 끼워 넣을 수 있는 상태이면 "약간의 추상(추한 모습)" 으로 지급률을 가산한 다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 137},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['eye', 'other']},
 'indexing': {'chunk_id': 'chunk_000877',
              'chunk_char_len': 126,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
