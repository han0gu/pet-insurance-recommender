from langchain_core.documents import Document

chunk = Document(
    page_content=('- 2. 강간 : 형법 제32장에서 정하는 강간죄\n'
 '- 3. 강도 : 형법 제38장에서 정하는 강도죄\n'
 '- 4. 상해, 폭행 및 폭력: 형법 제25장에서 정하는 상해와 폭행의 죄, 폭력행위 등 처벌\n'
 '- 에 관한 법률에 정한 폭력 등의 죄\n'
 '- ② 제1항에도 불구하고 제1항 제1호의 살인, 제4호의 「상해, 폭행 및 폭력」 등으로 피\n'
 '- 보험자의 신체에 피해가 발생한 경우에는 1개월을 초과하여 의사의 치료를 요하는 신\n'
 '- 체상해를 입은 때에만 보상합니다.'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000388',
              'chunk_char_len': 252,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
