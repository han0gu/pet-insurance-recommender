from langchain_core.documents import Document

chunk = Document(
    page_content=('| --- | --- | --- |\n'
 '| 관 보험료 납입을 연체하여 계약이 해지되고 계약자가 해약환급금을 받지 않은 경 우에 회사가 정하는 소정의 절차에 따라 해지된 계약을 '
 '다시 되살리는 일 | 관 보험료 납입을 연체하여 계약이 해지되고 계약자가 해약환급금을 받지 않은 경 우에 회사가 정하는 소정의 절차에 '
 '따라 해지된 계약을 다시 되살리는 일 | 관 보험료 납입을 연체하여 계약이 해지되고 계약자가 해약환급금을 받지 않은 경 우에 회사가 '
 '정하는 소정의 절차에 따라 해지된 계약을 다시 되살리는 일 |'),
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
