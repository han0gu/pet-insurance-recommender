from langchain_core.documents import Document

chunk = Document(
    page_content=('. 5) 순음청력검사를 실시하기 곤란하거나(청력의 감소가 의심되지만 의사소통이 되 지 않는 경우, 만 3세 미만의 소아 포함) 검사결과에 '
 '대한 검증이 필요한 경우 에는 "언어청력검사, 임피던스 청력검사, 청성뇌간반응검 사(ABR), 이음향방 사검사" 등을 추가실시 후 장해를 '
 '평가한다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 137},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000882',
              'chunk_char_len': 159,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.85}},
)
