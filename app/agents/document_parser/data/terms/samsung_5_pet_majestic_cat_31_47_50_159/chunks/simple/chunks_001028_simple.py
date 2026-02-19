from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 의료행위당시의 「건강보험 행위 급여·비급여 목록 및 급여 상대가치점수」에 따라 보험금 지급여부가 판단된 경우, 이후 수가코드가 '
 '변경되더라도 이 약관에서 보장하 는 의료행위 해당 여부를 다시 판단하지 않습니다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 157},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_001028',
              'chunk_char_len': 122,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
