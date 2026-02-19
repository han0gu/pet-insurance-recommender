from langchain_core.documents import Document

chunk = Document(
    page_content=('제 1조 (계약의 체결 및 효력)\n'
 '① 이 특별약관은 보험계약(특별약관이 부가된 경우에는 특별약관을 포함합니다. 이하 「보험계약」이라 합니다)을 체결 또는 변경할 때 다음 '
 '각 호의 경우 보험계약자(이하 「계약자」라 합니다)의 청약과 보험회사(이하「회사」라 합니다)의 승낙으로 계약에 부가하여 이루어집니다.\n'
 '1. 보험계약을 체결할 때 피보험자의 건강상태가 회사가 정한 기준에 적합하지 않은 경우 2. 보험계약을 체결한 후 계약 전 알릴 의무 '
 '위반의 효과 등으로 보장을 제한할 경우'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 129},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000818',
              'chunk_char_len': 266,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
