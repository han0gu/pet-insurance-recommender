from langchain_core.documents import Document

chunk = Document(
    page_content=('- 합니다) 중에 상해를 입고 그 직접적인 결과로써 [별표-상해관련3]5대골절 분류표에\n'
 '- 정한 골절(이하 「5대골절」 이라 합니다)을 입고 치료를 직접적인 목적으로 수술을 받\n'
 '- 은 경우에는 보험증권에 기재된 이 특별약관의 보험가입금액을 5대골절 수술비로 보\n'
 '- 험수익자에게 지급합니다.\n'
 '- ② 제1항의 5대골절 수술비는 매사고마다 각각 지급합니다. 다만, 동일한 상해사고를 직\n'
 '- 접적인 원인으로 동시에 2가지 이상 또는 2회 이상의 수술을 받은 경우에는 1회에 한\n'
 '- 하여 보상합니다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000345',
              'chunk_char_len': 274,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
