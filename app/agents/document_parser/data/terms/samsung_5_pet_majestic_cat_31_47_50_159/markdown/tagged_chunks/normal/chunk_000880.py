from langchain_core.documents import Document

chunk = Document(
    page_content=('수 및 산정지침)이 제9장(처치 및 수술료) 중 다음에 적은 수가코드에 해당하는 의료행위\n'
 '를 말하며, 이후에 보건복지부에서 고시하는 「건강보험 행위 급여·비급여 목록 및 급여\n'
 '상대가치점수」 개정에 따라 수가코드가 변경된 경우에는 개정된 기준을 적용합니다. 다\n'
 '만, 의료행위당시의 「건강보험 행위 급여·비급여 목록 및 급여 상대가치점수」에 따라\n'
 '보험금 지급여부가 판단된 경우, 이후 수가코드가 변경되더라도 이 약관에서 보장하는 의\n'
 '료행위 해당 여부를 다시 판단하지 않습니다.| 분류항목 | 분류항목 | 수가코드 |'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000880',
              'chunk_char_len': 286,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
