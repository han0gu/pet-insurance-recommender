from langchain_core.documents import Document

chunk = Document(
    page_content=('|  |  |  |\n'
 '| 3) 세안, 양치와 같은 개인위생관리를 독립적으로 시행 가능하나 목욕이나 샤워시 부분적으로 타인의 도움이 필요한 상태 | 3% |  '
 '|\n'
 '| 옷 입고 | 1) 상·하의 의복 착탈시 다른 사람의 계속적인 도움이 필요한 상태 | 10% |\n'
 '| 옷 입고 | 2) 상·하의 의복 착탈시 부분적으로 다른 사람의 도움이 필요한 상태 또는 상의 또는 하의중 하나만 혼자서 착 벗기 '
 '탈의가 가능한 상태 | 5% |'),
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
