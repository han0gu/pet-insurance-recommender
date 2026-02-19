from langchain_core.documents import Document

chunk = Document(
    page_content=('험률을 적용하여 산출된 보험료를 말합니다.\n'
 '2. 보험금감액법\n'
 '계약일부터 회사가 정하는 삭감기간 내에 보험계약의 규정에 정하는 상해 이외의 원인으로 보험계약의 보험금 지급사유가 발생하였을 경우에는 '
 '보험계약의 규정에 도 불구하고 계약을 체결할 때 정한 삭감기간에 따라 다음과 같이 보험금을 지급 합니다.\n'
 '경과기간 | 기준 | 삭감기간별 보험금지급비율\n'
 '1년 | 2년 | 3년 | 4년 | 5년\n'
 '1년미만 | 보험계약에 정한 지급보험금 | 50% | 30% | 25% | 20% | 15%'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 127},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000812',
              'chunk_char_len': 268,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
