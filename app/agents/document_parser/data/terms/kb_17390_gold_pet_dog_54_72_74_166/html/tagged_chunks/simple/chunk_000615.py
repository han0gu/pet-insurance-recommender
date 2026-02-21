from langchain_core.documents import Document

chunk = Document(
    page_content=("금쪽같은 펫보험(강아지)(무배당)(26.01) 87</p><br><p id='130' data-category='paragraph' "
 "style='font-size:18px'>- 87 -</p><p id='131' data-category='paragraph' "
 "style='font-size:14px'>제5조(특별약관의 소멸)<br>피보험자가 사망하였을 경우에는 이 특별약관의 계약도 소멸되며 "
 '회사는 "보험료 및</p><br><p id=\'132\' data-category=\'paragraph\''),
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
 'indexing': {'chunk_id': 'chunk_000615',
              'chunk_char_len': 274,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
