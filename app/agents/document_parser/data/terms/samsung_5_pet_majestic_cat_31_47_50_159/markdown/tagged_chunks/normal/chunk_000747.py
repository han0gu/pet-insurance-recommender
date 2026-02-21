from langchain_core.documents import Document

chunk = Document(
    page_content=('말한다.| 항목 | 내 용 | 점수 |\n'
 '| --- | --- | --- |\n'
 '| 검사 소견 | 양측 전정기능 소실 | 14 |\n'
 '| 검사 소견 | 양측 전정기능 감소 | 10 |\n'
 '| 검사 소견 | 일측 전정기능 소실 | 4 |\n'
 '| 치료 병력 | 장기 통원치료(1년간 12회이상) | 6 |\n'
 '| 치료 병력 | 장기 통원치료(1년간 6회이상) | 4 |\n'
 '| 치료 병력 | 단기 통원치료(6개월간 6회이상) | 2 |\n'
 '| 치료 병력 | 단기 통원치료(6개월간 6회미만) | 0 |'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000747',
              'chunk_char_len': 263,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
