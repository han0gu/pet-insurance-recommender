from langchain_core.documents import Document

chunk = Document(
    page_content=("id='125' data-category='paragraph' style='font-size:16px'>제3조(해지된 특별약관의 "
 "부활(효력회복))</p><br><h1 id='126' style='font-size:16px'>회사는 이 특약의 "
 "부활(효력회복)</h1><br><p id='127' data-category='paragraph' "
 "style='font-size:16px'>을 승낙한 경우에 한하여 제4장 반려동물 관련 특별약관 반려동물(강아지) "
 '일반조항<br>제17조(보험료의 납입을 연체하여 해지된 특별약관의'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_001456',
              'chunk_char_len': 290,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
