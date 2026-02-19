from langchain_core.documents import Document

chunk = Document(
    page_content=('① 제29조(보험료의 납입이 연체되는 경우 납입최고(독촉)와 특별약관의 해지)에 따라 계 약이 해지되었으나 해약환급금을 받지 않은 '
 '경우(보험계약대출 등에 따라 해약환급금 이 차감되었으나 받지 않은 경우 또는 해약환급금이 없는 경우를 포함합니다) 계약자 는 해지된 '
 '날부터 3년 이내에 회사가 정한 절차에 따라 계약의 부활(효력회복)을 청약 할 수 있습니다'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 62},
 'term_type': 'special',
 'clause': {'clause_type': 'renewal', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000292',
              'chunk_char_len': 197,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
