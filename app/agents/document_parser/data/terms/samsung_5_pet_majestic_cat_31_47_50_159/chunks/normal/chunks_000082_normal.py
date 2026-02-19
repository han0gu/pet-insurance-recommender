from langchain_core.documents import Document

chunk = Document(
    page_content=('제4관 보험계약의 성립과 유지\n'
 '제 20조 (보험계약의 성립)\n'
 '① 계약은 계약자의 청약과 회사의 승낙으로 이루어집니다. ② 회사는 피보험자가 계약에 적합하지 않은 경우에는 승낙을 거절하거나 별도의 '
 '조건(보 험가입금액 제한, 일부보장 제외, 보험금 삭감, 보험료 할증 등)을 붙여 승낙할 수 있 습니다.\n'
 '<용어풀이>\n'
 '[보험가입금액 제한]'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 38},
 'term_type': 'basic',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000082',
              'chunk_char_len': 186,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
