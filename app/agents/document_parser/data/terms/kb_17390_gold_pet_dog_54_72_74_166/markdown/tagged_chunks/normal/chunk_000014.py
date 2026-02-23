from langchain_core.documents import Document

chunk = Document(
    page_content=('- 별\n'
 '- 2. 제2조제4호 또는 제9호의 공휴일이 일요일과 겹치는 경우\n'
 '- 표\n'
 '- 3. 제2조제2호ㆍ제4호ㆍ제7호 또는 제9호의 공휴일이 토요일ㆍ일요일이\n'
 '- 아닌 날에 같은 조 제2호부터 제10호까지의 규정에 따른 다른 공휴일\n'
 '- 과 겹치는 경우\n'
 '② 제1항에 따른 대체공휴일이 같은 날에 겹치는 경우에는 그 대체공휴일 다음- 의 첫 번째 비공휴일까지 대체공휴일로 한다.\n'
 '- 법\n'
 '- ③ 제1항 및 제2항에 따른 대체공휴일이 토요일인 경우에는 그 다음의 첫 번째 ㆍ\n'
 '- 비공휴일을 대체공휴일로 한다. 규정'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000014',
              'chunk_char_len': 281,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
