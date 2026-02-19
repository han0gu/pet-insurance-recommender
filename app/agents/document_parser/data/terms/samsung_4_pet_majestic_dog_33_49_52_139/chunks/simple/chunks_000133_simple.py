from langchain_core.documents import Document

chunk = Document(
    page_content=('. 계약자가 회사로부터 보험계약대출을 받은 경우 계약이 해지되는 즉시 해약환급금에 서 보험계약대출원금과 이자가 차감된다는 내용 ② '
 '납입최고(독촉)기간의 마지막 날이 영업일이 아닌 때에는 최고(독촉)기간은 그 다음 날까지로 합니다. ③ 보험수익자와 계약자가 다른 경우 '
 '보험수익자에게도 제1항에 따른 내용을 알려 드립 니다. ④ 보험료 납입이 연체중이라도 계약의 해지 전에 발생한 보험금 지급사유에 대하여 '
 '회 사는 보상하여 드립니다'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 45},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000133',
              'chunk_char_len': 239,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
