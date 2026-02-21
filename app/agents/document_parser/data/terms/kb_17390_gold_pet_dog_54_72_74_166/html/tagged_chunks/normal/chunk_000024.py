from langchain_core.documents import Document

chunk = Document(
    page_content=(". 이하 같다)을 대체공휴일로 한다.</p><br><p id='23' data-category='list' "
 "style='font-size:14px'>1. 제2조제2호 또는 제7호의 공휴일이 토요일이나 일요일과 겹치는 경우<br>별<br>2. "
 '제2조제4호 또는 제9호의 공휴일이 일요일과 겹치는 경우<br>표<br>3'),
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
 'indexing': {'chunk_id': 'chunk_000024',
              'chunk_char_len': 176,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
