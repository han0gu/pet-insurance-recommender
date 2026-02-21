from langchain_core.documents import Document

chunk = Document(
    page_content=("경우</p><br><p id='137' data-category='paragraph' style='font-size:16px'>② "
 "제1항에 따른 대체공휴일이 같은 날에 겹치는 경우에는 그 대체공휴일 다</p><p id='138' "
 "data-category='paragraph' style='font-size:16px'>※</p><br><p id='139' "
 "data-category='list' style='font-size:16px'>음의 첫 번째 비공휴일까지 대체공휴일로 "
 '한다.<br>반<br>③ 제1항 및 제2항에 따른 대체공휴일이'),
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
 'indexing': {'chunk_id': 'chunk_000778',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
