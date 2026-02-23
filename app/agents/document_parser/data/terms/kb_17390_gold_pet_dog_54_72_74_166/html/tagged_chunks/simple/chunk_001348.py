from langchain_core.documents import Document

chunk = Document(
    page_content=('청구)<br>\uf000 피보험자 또는 지정대리청구인은 제1조에 정한 특별약관의 보험기간 중에 회사가<br>정하는 바에 따라 다음의 '
 "서류를 제출하고 이 특별약관의 보험금을 청구하여야 합</p><br><p id='225' data-category='list' "
 "style='font-size:14px'>니다.<br>1"),
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
 'indexing': {'chunk_id': 'chunk_001348',
              'chunk_char_len': 170,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
