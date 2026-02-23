from langchain_core.documents import Document

chunk = Document(
    page_content=('. 제2조제2호ㆍ제4호ㆍ제7호 또는 제9호의 공휴일이 토요일ㆍ일요일이<br>아닌 날에 같은 조 제2호부터 제10호까지의 규정에 따른 다른 '
 "공휴일<br>과 겹치는 경우</p><br><p id='24' data-category='paragraph' "
 "style='font-size:16px'>② 제1항에 따른 대체공휴일이 같은 날에 겹치는 경우에는 그 대체공휴일 다음</p><br><p "
 "id='25' data-category='list' style='font-size:16px'>의 첫 번째 비공휴일까지 대체공휴일로 "
 '한다.<br>법<br>③'),
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
 'indexing': {'chunk_id': 'chunk_000025',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
