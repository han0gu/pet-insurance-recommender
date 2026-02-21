from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:18px'>- 108 -</p><p id='108' data-category='paragraph' "
 "style='font-size:16px'>제34조</p><br><p id='109' data-category='paragraph' "
 "style='font-size:16px'>제1항에 따른 해약환급금을 계약자에게 지급합니다.</p><br><p id='110' "
 "data-category='paragraph' style='font-size:16px'>제23조(준용규정)<br>반려동물(강아지) "
 '일반조항에서 정하지'),
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
 'indexing': {'chunk_id': 'chunk_000930',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
