from langchain_core.documents import Document

chunk = Document(
    page_content=('rowspan="4">유형 이동동작</td><td>제한 정도 지급률 1) 특별한 보조기구를 사용함에도 불구하고 다른 사람의 계속적인 '
 '도움이 없이는 방 밖을 나올 수 없는 상태 40% 또는 침대에서 휠체어로 옮기기를 포함하여 휠체어 이 다른 필요한 '
 '상태</td><td>사항</td></tr><tr><td>동시 사람의 계속적인 도움이 2) 휠체어 또는 다른 사람의 도움 없이는 방밖을 '
 '나올 수 없는 상태 또는 보행이 불가능하나 스스로 휠체어를 30% 밀어 이동이 가능한 상태</td><td>보 통약 '
 '관</td></tr><tr><td>3)'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
