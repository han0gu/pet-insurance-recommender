from langchain_core.documents import Document

chunk = Document(
    page_content=('- 르며, 이 특별약관 체결 시점 이후 보건복지부에서 고시하는 「건강보험 행위 급여·비\n'
 '- 급여 목록 및 급여 상대가치점수」 개정에 따라 수가코드가 변경된 경우에는 개정된\n'
 '- 기준을 적용합니다. 다만, 「건강보험 행위 급여·비급여 목록 및 급여 상대가치점수」\n'
 '- 가 폐지되어 보험금 지급사유에 대한 판정이 불가능한 경우 폐지 직전의 관련 법규에\n'
 '- 서 정한 분류번호 및 코드를 따릅니다.\n'
 '- ⑥ 제5항에도 불구하고 이 특별약관에서 보장하는 창상봉합술 시행 당시의 「건강보험'),
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
 'indexing': {'chunk_id': 'chunk_000438',
              'chunk_char_len': 264,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
