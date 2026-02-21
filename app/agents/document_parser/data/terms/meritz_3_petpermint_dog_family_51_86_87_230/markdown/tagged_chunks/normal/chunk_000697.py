from langchain_core.documents import Document

chunk = Document(
    page_content=('| 이동 동작 | - 특별한 보조기구를 사용함에도 불구하고 다른 사람의 계속적인 도움이 없이는 방밖을 나올 수 없는 상태 또는 침대에서 '
 '휠체어로 옮기기 를 포함하여 휠체어 이동시 다른 사람의 계속 적인 도움이 필요한 상태(지급률 40%) - 휠체어 또는 다른 사람의 '
 '도움없이는 방밖을 나올 수 없는 상태 또는 보행이 불가능하나 스 스로 휠체어를 밀어 이동이 가능한 상태(30%) - 목발 또는 '
 '보행기(walker)를 사용하지 않으면 독립적인 보행이 불가능한 상태(20%) - 보조기구 없이 독립적인 보행은 가능하나 보행 시'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000697',
              'chunk_char_len': 292,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
