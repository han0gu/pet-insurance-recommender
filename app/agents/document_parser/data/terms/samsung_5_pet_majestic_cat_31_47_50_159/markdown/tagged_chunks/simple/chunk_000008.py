from langchain_core.documents import Document

chunk = Document(
    page_content=('- - 보장보험료 = 위험보험료 + 부가보험료\n'
 '- - 적립보험료 = 적립부분 순보험료 + 부가보험료\n'
 '⑥ [갱신형] 특별약관의 갱신 관련 용어- 1. 최초계약: [갱신형] 특별약관이 최초로 부가되는 경우를 말합니다.\n'
 '- 2. 갱신계약: [갱신형] 특별약관의 보험기간이 끝난 후 제도성 특별약관 「5-1. [갱신\n'
 '- 31 -형] 특별약관의 자동갱신 특별약관」에 따라 갱신된 경우를 말합니다.3. 갱신일: [갱신형] 특별약관이 갱신되기 직전 '
 '계약(이하「갱신 전 계약」이라 합니'),
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
 'indexing': {'chunk_id': 'chunk_000008',
              'chunk_char_len': 263,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
