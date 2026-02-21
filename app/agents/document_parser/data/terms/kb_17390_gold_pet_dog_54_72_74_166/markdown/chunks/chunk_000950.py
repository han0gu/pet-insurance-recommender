from langchain_core.documents import Document

chunk = Document(
    page_content=('| 배변· | 4) 빈번하고 불규칙한 배변으로 인해 2시간 이상 계속되 는 업무를 수행하는 것이 어려운 상태, 또는 배변, 배 뇨는 '
 '독립적으로 가능하나 요실금, 변실금이 있는 때 | 5% |\n'
 '|  | 1) 세안, 양치, 샤워, 목욕 등 모든 개인위생 관리시 타 인의 지속적인 도움이 필요한 상태 | 10% |\n'
 '|  | 2) 세안, 양치시 부분적인 도움 하에 혼자서 가능하나 목 목욕 욕이나 샤워시 타인의 도움이 필요한 상태 | 5% |\n'
 '|  |  |  |'),
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
