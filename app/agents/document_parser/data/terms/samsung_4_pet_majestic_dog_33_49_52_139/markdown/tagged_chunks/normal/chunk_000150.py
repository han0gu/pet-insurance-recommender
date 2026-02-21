from langchain_core.documents import Document

chunk = Document(
    page_content=('든 예금보호 대상 금융상품의 해약환급금(또는 만기 시 보험금)에 기타 지급금을 합한 금액이 1인- 48 -당 "1억원까지" 예금자 보호가 '
 '됩니다. 이와 별도로 본 회사 보호상품의 사고보험금을 합산한 금액\n'
 '이 1인당 "1억원까지" 보호됩니다. 다만, 보험계약자 및 보험료 납부자가 법인인 보험계약의 경우\n'
 '에는 보호되지 않습니다.- 49 -※ 약관에서 인용된 법·규정은「별표 및 참고」의 「약관에서 인용된 법·규정」에서'),
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
 'indexing': {'chunk_id': 'chunk_000150',
              'chunk_char_len': 232,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
