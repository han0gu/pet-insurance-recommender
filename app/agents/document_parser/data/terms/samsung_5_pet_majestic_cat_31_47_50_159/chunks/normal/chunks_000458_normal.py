from langchain_core.documents import Document

chunk = Document(
    page_content=('① 이 특별약관에서 강력범죄는 아래의 항목에 해당하는 죄를 말합니다.\n'
 '1. 살인 : 형법 제24장에서 정하는 살인죄 2. 강간 : 형법 제32장에서 정하는 강간죄 3. 강도 : 형법 제38장에서 정하는 '
 '강도죄 4. 상해, 폭행 및 폭력: 형법 제25장에서 정하는 상해와 폭행의 죄, 폭력행위 등 처벌 에 관한 법률에 정한 폭력 등의 죄'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 85},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000458',
              'chunk_char_len': 187,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
