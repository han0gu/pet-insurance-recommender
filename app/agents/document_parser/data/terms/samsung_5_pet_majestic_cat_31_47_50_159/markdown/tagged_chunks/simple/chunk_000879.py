from langchain_core.documents import Document

chunk = Document(
    page_content=('| 안면과 경부 이외 | 창상봉합술(안면과경부이외,변연절제포함,근육,길이10cm이상, 10cm마다 추가) | SC040 |\n'
 '- 154 -[별표-상해및질병관련2] 급여 창상봉합술(안면부) 대상 수가코드약 관에 규정하는 창상봉합술(안면부, 급여)로 분류되는 치료는 '
 '보건복지부가 고시하는 건\n'
 '강보험 행위 급여·비급여 목록 및 급여 상대가치점수」제2부(행위 급여 목록·상대가치점\n'
 '수 및 산정지침)이 제9장(처치 및 수술료) 중 다음에 적은 수가코드에 해당하는 의료행위'),
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
 'indexing': {'chunk_id': 'chunk_000879',
              'chunk_char_len': 257,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
