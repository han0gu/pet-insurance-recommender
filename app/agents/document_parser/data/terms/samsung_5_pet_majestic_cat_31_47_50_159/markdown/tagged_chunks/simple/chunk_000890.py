from langchain_core.documents import Document

chunk = Document(
    page_content=('| 안면 또는 경부 | 창상봉합술(안면또는경부,변연절제포함,근육,길이7.5cm이상~10.0cm미만) | SA039 |\n'
 '| 안면 또는 경부 | 창상봉합술(안면또는경부,변연절제포함,근육,길이10cm 이상, 5cm마다 추가) | SA040 |\n'
 '- 156 -# [별표-상해및질병관련4] 급여 창상봉합술(3/5cm미만) 대상 수가코드약 관에 규정하는 창상봉합술(3/5cm미만, '
 '급여)로 분류되는 치료는 보건복지부가 고시하\n'
 '는 건강보험 행위 급여·비급여 목록 및 급여 상대가치점수」제2부(행위 급여 목록·상대가'),
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
 'indexing': {'chunk_id': 'chunk_000890',
              'chunk_char_len': 279,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
