from langchain_core.documents import Document

chunk = Document(
    page_content=('① 이 약관에 따른 해약환급금은 “보험료 및 해약환급금 산출방법서”에 따라 계산합니 다. 이 때 적립부분 순보험료에 대하여는 보험료 '
 '납입일(회사에 입금된 날을 말합니 다)부터 공시이율을 적용하고, 제10조(환급금의 중도인출)에 따라 중도인출한 경우에 는 “보험료 및 '
 '해약환급금 산출방법서”에 따라 적립부분 계약자적립액에서 중도인 출한 금액을 차감하여 계산합니다. 다만, 보험기간 중에 공시이율이 변경되는 '
 '경우에 는 변경된 시점 이후부터 제9조(공시이율의 적용 및 공시) 제1항에 따라 변경된 이율 을 적용합니다'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 46},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000155',
              'chunk_char_len': 285,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
