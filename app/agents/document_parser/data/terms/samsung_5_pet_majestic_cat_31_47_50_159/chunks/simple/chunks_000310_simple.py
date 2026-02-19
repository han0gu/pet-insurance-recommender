from langchain_core.documents import Document

chunk = Document(
    page_content=('제 35조 (해약환급금)\n'
 '① 이 약관에 따른 해약환급금은 “보험료 및 해약환급금 산출방법서”에 따라 계산하며, 계약이 해지될 경우에는 아래와 같이 해약환급금을 '
 '지급합니다.\n'
 '1. 해약환급금 구분이 해약환급금 일부지급형일 때에는 보험료 납입기간 중 계약이 해지될 경우 표준형 상품 해약환급금의 50%에 해당하는 '
 '금액을 지급하며, 보험료 납입이 완료되고 보험료 납입기간이 종료된 이후 계약이 해지될 경우 표준형 상품 해약환급금의 100%에 해당하는 '
 '금액을 지급합니다.\n'
 '<유의사항>\n'
 '[해약환급금 일부지급형의 해약환급금 관련]'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 62},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000310',
              'chunk_char_len': 289,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
