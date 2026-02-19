from langchain_core.documents import Document

chunk = Document(
    page_content=('약 관에 규정하는 창상봉합술(급여)로 분류되는 치료는 보건복지부가 고시하는 건강보험 행위 급여·비급여 목록 및 급여 '
 '상대가치점수」제2부(행위 급여 목록·상대가치점수 및 산 정지침)이 제9장(처치 및 수술료) 중 다음에 적은 수가코드에 해당하는 의료행위를 '
 '말하 며, 이후에 보건복지부에서 고시하는 「건강보험 행위 급여·비급여 목록 및 급여 상대가 치점수」 개정에 따라 수가코드가 변경된 '
 '경우에는 개정된 기준을 적용합니다'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 154},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_001006',
              'chunk_char_len': 233,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
