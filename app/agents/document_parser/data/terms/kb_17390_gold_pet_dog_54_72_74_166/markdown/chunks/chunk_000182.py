from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 회사가 제1항에 따라 계약을 해지한 경우 회사는 그 취지를 계약자에게 통지하고\n'
 '제34조(해약환급금) 제1항에 따른 해약환급금을 지급합니다.제33조(회사의 파산선고와 해지)\uf000 회사가 파산의 선고를 받은 '
 '때에는계약자는 계약을 해지할 수 있습니다.\uf000 제1항에 따라 해지하지 않은 계약은 파산선고 후 3개월이 지난 때에는 그 효력을\n'
 '잃습니다.\n'
 '\uf000 제1항에 따라 계약이 해지되거나 제2항에 따라 계약이 효력을 잃는 경우에 회사는\n'
 '제34조(해약환급금) 제1항에 의한 해약환급금을 계약자에게 지급합니다.-'),
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
