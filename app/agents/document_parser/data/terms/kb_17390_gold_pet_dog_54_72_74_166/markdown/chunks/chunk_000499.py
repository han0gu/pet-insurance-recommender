from langchain_core.documents import Document

chunk = Document(
    page_content=('| 현재 | 있는 소멸시키 |\n'
 '| 유지되고 계약 또는 효력이 상실된 계약을 장래를 향하여 거나 계약유지 의사를 포기하여 만기일 이전에 계약관계를 청산하는 것 | '
 '유지되고 계약 또는 효력이 상실된 계약을 장래를 향하여 거나 계약유지 의사를 포기하여 만기일 이전에 계약관계를 청산하는 것 |\n'
 '제10조(사기에 의한 계약)# \uf000 계약자 또는피보험자의 사기에 의하여 계약이 성립되었음을 회사가 증명하는 경- 우에는 계약일부터 '
 '5년 이내(사기사실을 안 날부터 1개월 이내)에 계약을 취소할\n'
 '- 수 있습니다.'),
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
