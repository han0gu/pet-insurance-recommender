from langchain_core.documents import Document

chunk = Document(
    page_content=('가) 해당 관절의 운동범위 합계가 정상 운동범위의 3/4 이하로 제한된 경우 나) 근전도 검사상 불완전한 손상(incomplete '
 'injury)소견이 있으면서 도수근력 검사(MMT)에서 근력이 "3등급(fair)" 인 경우'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 143},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['joint']},
 'indexing': {'chunk_id': 'chunk_000933',
              'chunk_char_len': 124,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
