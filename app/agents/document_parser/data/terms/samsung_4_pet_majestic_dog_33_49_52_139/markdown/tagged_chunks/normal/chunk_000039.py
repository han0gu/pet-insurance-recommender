from langchain_core.documents import Document

chunk = Document(
    page_content=('금된 날을 말합니다)부터 공시이율로 “보험료 및 해약환급금 산출방법서”에 따라\n'
 '적립한 금액(제10조(환급금의 중도인출)에 따라 중도인출한 경우에는 중도인출한 금\n'
 '액을 차감하고 적립한 금액을 말합니다)을 만기환급금으로 보험수익자에게 지급합니\n'
 '다. 다만, 보험기간 중에 공시이율이 변경되는 경우에는 변경된 시점 이후부터 제9조\n'
 '(공시이율의 적용 및 공시) 제1항에 따라 변경된 이율을 적용하며, 최저보증이율은 연'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000039',
              'chunk_char_len': 228,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
