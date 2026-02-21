from langchain_core.documents import Document

chunk = Document(
    page_content=('| --- | --- | --- |\n'
 '| 유형 이동동작 | 제한 정도 지급률 1) 특별한 보조기구를 사용함에도 불구하고 다른 사람의 계속적인 도움이 없이는 방 밖을 나올 수 '
 '없는 상태 40% 또는 침대에서 휠체어로 옮기기를 포함하여 휠체어 이 동시 다른 사람의 계속적인 필요한 상태 | 사항 |\n'
 '| 유형 이동동작 | 도움이 2) 휠체어 또는 다른 사람의 도움 없이는 방밖을 나올 수 없는 상태 또는 보행이 불가능하나 스스로 휠체어를 '
 '30% 밀어 이동이 가능한 상태 | 보 통약 관 |'),
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
 'indexing': {'chunk_id': 'chunk_000945',
              'chunk_char_len': 269,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
