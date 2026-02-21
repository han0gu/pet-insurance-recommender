from langchain_core.documents import Document

chunk = Document(
    page_content=('| 이동 동작 | · 특별한 보조기구를 사용함에도 불구하고 다른 사람의 계속적인 도움이 없이는 방 밖을 나올 수 없는 상태 또는 침대에서 '
 '휠체어로 옮기기를 포함하여 휠체어 이동 시 다른 사람의 계속적인 도움이 필요한 상태(40%) · 휠체어 또는 다른 사람의 도움 없이는 '
 '방밖을 나올 수 없는 상태 또는 보행이 불 가능하나 스스로 휠체어를 밀어 이동이 가능한 상태(30%) · 목발 또는 '
 '보행기(walker)를 사용하지 않으면 독립적인 보행이 불가능한 상태 (20%) · 보조기구 없이 독립적인 보행은 가능하나 보행시 '
 '파행(절뚝거림)이'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
