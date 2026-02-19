from langchain_core.documents import Document

chunk = Document(
    page_content=('① 회사는 보험기간이 끝난 때에는 적립부분 순보험료에 대하여 보험료납입일(회사에 입 금된 날을 말합니다)부터 공시이율로 “보험료 및 '
 '해약환급금 산출방법서”에 따라 적립한 금액(제10조(환급금의 중도인출)에 따라 중도인출한 경우에는 중도인출한 금 액을 차감하고 적립한 '
 '금액을 말합니다)을 만기환급금으로 보험수익자에게 지급합니 다. 다만, 보험기간 중에 공시이율이 변경되는 경우에는 변경된 시점 이후부터 '
 '제9조 (공시이율의 적용 및 공시) 제1항에 따라 변경된 이율을 적용하며, 최저보증이율은 연 단위 복리 0.25%로 합니다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 35},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000044',
              'chunk_char_len': 293,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
