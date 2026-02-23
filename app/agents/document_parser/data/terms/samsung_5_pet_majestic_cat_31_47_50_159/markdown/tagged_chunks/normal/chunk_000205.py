from langchain_core.documents import Document

chunk = Document(
    page_content=('- ② 제1항의 규정에도 불구하고 기본계약의 보장개시일 이후에 이 특별약관을 청약하는\n'
 '- 경우에는 회사의 승낙을 얻어 기본계약에 부가하여 이 특별약관을 체결할 수 있습니\n'
 '- 다.\n'
 '- ③ 회사는 피보험자가 이 특별약관에 적합하지 않은 경우에는 승낙을 거절하거나 별도의\n'
 '- 조건(보험가입금액 제한, 일부보장 제외, 보험금 삭감, 보험료 할증 등)을 붙여 승낙할\n'
 '- 수 있습니다.\n'
 '<용어풀이># [보험가입금액 제한]피보험자가 가입을 할 수 있는 최대 보험가입금액을 제한하는 방법을 말합니다.\n'
 '[일부보장 제외]'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000205',
              'chunk_char_len': 281,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
