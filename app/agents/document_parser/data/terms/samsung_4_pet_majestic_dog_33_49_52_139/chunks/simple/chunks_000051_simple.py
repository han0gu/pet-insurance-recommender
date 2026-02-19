from langchain_core.documents import Document

chunk = Document(
    page_content=('② 회사는 제1항에 따라 일시에 지급할 금액을 나누어 지급하는 경우에는 나중에 지급할 금액에 대하여 평균공시이율을 연단위 복리로 계산한 '
 '금액을 더하며, 나누어 지급할 금액을 일시에 지급하는 경우에는 평균공시이율을 연단위 복리로 할인한 금액을 지급 합니다. ③ 제2항에도 '
 '불구하고 회사는 「보험금의 지급사유」에서 정한 나누어 지급하는 보험금 에 대해서 일시에 지급하는 경우에 한하여 평균공시이율을 반영하여 '
 '연단위 복리로 할인한 금액과 보장부분 적용이율을 반영하여 연단위 복리로 할인한 금액 중 큰 금액 을 지급합니다.\n'
 '<예시안내>'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 37},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000051',
              'chunk_char_len': 295,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
