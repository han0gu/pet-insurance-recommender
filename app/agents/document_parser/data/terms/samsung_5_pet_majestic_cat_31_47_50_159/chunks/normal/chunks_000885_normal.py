from langchain_core.documents import Document

chunk = Document(
    page_content=('양측 전정기능 감소 | 10\n'
 '일측 전정기능 소실 | 4\n'
 '치료 병력 | 장기 통원치료(1년간 12회이상) | 6\n'
 '장기 통원치료(1년간 6회이상) | 4\n'
 '단기 통원치료(6개월간 6회이상) | 2\n'
 '단기 통원치료(6개월간 6회미만) | 0\n'
 '기능 장해 소견 | 두 눈을 감고 일어서기 곤란하거나 두 눈을 뜨고 10m 거리를 직선으 로 걷다가 쓰러지는 경우 | 20'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 138},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['head', 'other']},
 'indexing': {'chunk_id': 'chunk_000885',
              'chunk_char_len': 198,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
