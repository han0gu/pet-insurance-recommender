from langchain_core.documents import Document

chunk = Document(
    page_content=('| --- | --- | --- |\n'
 '| 해당 보험 상품의 해약환급금 범위 내에서 납입할 보험료를 자동적으로 대출하 여 이를 보험료 납입에 충당하는 서비스를 말합니다. | '
 '해당 보험 상품의 해약환급금 범위 내에서 납입할 보험료를 자동적으로 대출하 여 이를 보험료 납입에 충당하는 서비스를 말합니다. | 해당 '
 '보험 상품의 해약환급금 범위 내에서 납입할 보험료를 자동적으로 대출하 여 이를 보험료 납입에 충당하는 서비스를 말합니다. |'),
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
