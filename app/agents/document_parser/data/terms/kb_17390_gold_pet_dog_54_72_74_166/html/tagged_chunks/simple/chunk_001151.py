from langchain_core.documents import Document

chunk = Document(
    page_content=('. 피보험자와 세대를 같이하는 친족에 대한 배상책임<br>해(이하 "배상책임손해"라 합니다)를 이 특별약관에 따라 보상하여 '
 '드립니다.<br>4. 피보험자가 소유, 사용 또는 관리하는 재물이 손해를 입었을 경우에 그 재물<br>에 대하여 정당한 권리를 가진 '
 '사람에게 부담하는 손해에 대한 배상책임.<br>제4조(보상하는 손해의 범위)<br>5. 피보험자의 심신상실로 인한 배상책임<br>회사가 '
 '1사고당 보상하는 손해의 범위는 아래와 같습니다.<br>6. 피보험자의 지시에 따른 배상책임<br>1'),
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
 'indexing': {'chunk_id': 'chunk_001151',
              'chunk_char_len': 275,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
