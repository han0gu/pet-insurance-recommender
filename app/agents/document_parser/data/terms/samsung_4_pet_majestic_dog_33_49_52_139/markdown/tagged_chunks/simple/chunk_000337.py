from langchain_core.documents import Document

chunk = Document(
    page_content=('합니다) 중에 상해를 입고 그 직접적인 결과로써 [별표-상해관련1]골절 분류표에 정\n'
 '한 골절(이하 「골절」 이라 합니다)을 입고 치료를 직접적인 목적으로 수술을 받은 경\n'
 '우에는 보험증권에 기재된 이 특별약관의 보험가입금액을 상해 골절 수술비로 보험수\n'
 '익자에게 지급합니다.\n'
 '② 제1항의 상해 골절 수술비는 매사고마다 지급합니다. 다만, 동일한 상해사고를 직접적\n'
 '인 원인으로 동시에 2가지 이상 또는 2회 이상의 수술을 받은 경우에는 1회에 한하여\n'
 '보상합니다.-'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000337',
              'chunk_char_len': 256,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
