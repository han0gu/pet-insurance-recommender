from langchain_core.documents import Document

chunk = Document(
    page_content=('- ⑥ 제5항에도 불구하고 이 특별약관에서 보장하는 창상봉합술 시행 당시의 「건강보험\n'
 '- 행위 급여·비급여 목록 및 급여 상대가치점수」 에 따라 보험금 지급여부가 판단된 경\n'
 '- 우, 이후 「건강보험 행위 급여·비급여 목록 및 급여 상대가치점수」 개정으로 급여치\n'
 '- 료 판정이 변경되더라도 이 특별약관에서 보장하는 창상봉합술 해당 여부를 다시 판\n'
 '- 단하지 않습니다.\n'
 '- ⑦ 제1항 내지 제4항의 창상봉합술은 「국민건강보험법」 에서 정한 요양급여 또는 「의'),
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
 'indexing': {'chunk_id': 'chunk_000439',
              'chunk_char_len': 254,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
