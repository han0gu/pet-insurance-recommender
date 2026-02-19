from langchain_core.documents import Document

chunk = Document(
    page_content=('지될 경우 해약환급금을 지급하지 않으며, 보험료 납입이 완료되고 보험료 납입기 간이 종료된 이후 계약이 해지될 경우 표준형 상품의 '
 '해약환급률에 이 상품의 해 지 시점까지 납입한 보험료를 곱한 금액을 지급합니다. 이 때, 표준형 상품의 해약 환급률이란 표준형 상품의 '
 '해약환급금을 표준형 상품의 해지 시점까지 납입한 보 험료로 나눈 비율을 말하며, 해지 시점까지 납입한 보험료란 보험가입금액의 감액 등 '
 '변경사항을 반영하여 계산한 해지 시점의 보험료에 해지 시점까지의 납입회차 를 곱한 금액을 말합니다.\n'
 '<유의사항>'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 64},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['eye']},
 'indexing': {'chunk_id': 'chunk_000313',
              'chunk_char_len': 285,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
