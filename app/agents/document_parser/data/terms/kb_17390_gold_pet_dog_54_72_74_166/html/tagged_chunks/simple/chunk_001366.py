from langchain_core.documents import Document

chunk = Document(
    page_content=(". 보장특약 자동갱신(추가납입형)</h1><p id='13' data-category='paragraph' "
 'style=\'font-size:14px\'>제1조(특약의 적용)<br>이 "보장특약 자동갱신(추가납입형) 특별약관"(이하 "특약"이라 '
 '합니다)은 손해의<br>보상을 내용으로 한 이 계약의 특별약관(이하 "보장특약"이라 합니다)의 자동갱신에</p><br><h1 '
 "id='14' style='font-size:14px'>대하여 회사와 계약자간에 사전에 합의가 있을 경우에 적용합니다.</h1><p "
 "id='15'"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_001366',
              'chunk_char_len': 285,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
