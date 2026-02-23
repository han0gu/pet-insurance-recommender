from langchain_core.documents import Document

chunk = Document(
    page_content=('- 초과할 때에 한하여 그 초과액만을 보상합니다. 다만, 의무보험이 다수인 경우에는 제\n'
 '- 10조(보험금의 분담)를 따릅니다.\n'
 '- ② 제1항의 의무보험은 피보험자가 법률에 의하여 의무적으로 가입하여야 하는 보험으로\n'
 '- 서 공제계약을 포함합니다.\n'
 '- ③ 피보험자가 의무보험에 가입하여야 함에도 불구하고 가입하지 않은 경우에는 그가 가\n'
 '- 입했더라면 의무보험에서 보상했을 금액을 제1항의 “의무보험에서 보상하는 금액”\n'
 '- 으로 봅니다.'),
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
 'indexing': {'chunk_id': 'chunk_000646',
              'chunk_char_len': 241,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
