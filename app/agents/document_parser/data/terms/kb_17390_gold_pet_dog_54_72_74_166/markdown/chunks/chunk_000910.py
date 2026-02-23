from langchain_core.documents import Document

chunk = Document(
    page_content=('한다.- \n'
 '다.- 지급률의 결정\n'
 '- 1) 한 다리의 3대 관절 중 관절 하나에 기능장해가 생기고 다른 관절 하나 특별\n'
 '- 에 기능장해가 발생한 경우 지급률은 각각 적용하여 합산한다. 약\n'
 '- 2) 1하지(다리와 발가락)의 후유장해 지급률은 원칙적으로 각각 합산하되, 관\n'
 '- 지급률은 60% 한도로 한다.\n'
 '10. 손가락의 장해- \n'
 '가. 장해의 분류별장해의 분류 지급률\n'
 '1) 한 손의 5개 손가락을 모두 잃었을 때 55\n'
 '2) 한 손의 첫째 손가락을 잃었을 때 15\n'
 '3) 한 손의 첫째 손가락 이외의 손가락을 잃었을 때 법\n'
 '10 ㆍ'),
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
