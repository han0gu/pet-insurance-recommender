from langchain_core.documents import Document

chunk = Document(
    page_content=('② 납입최고(독촉)기간의 마지막 날이 영업일이 아닌 때에는 최고(독촉)기간은 그 다음 날까지로 합니다. ③ 보험수익자와 계약자가 다른 '
 '경우 보험수익자에게도 제1항에 따른 내용을 알려 드립 니다. ④ 보험료 납입이 연체중이라도 계약의 해지 전에 발생한 보험금 지급사유에 '
 '대하여 회 사는 보상하며, 계약의 해지 전에 보험료 납입면제 사유가 발생한 경우 회사는 계약을 해지하지 않고 차회 이후의 보험료 납입을 '
 '면제하여 드립니다'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 60},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000286',
              'chunk_char_len': 235,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
