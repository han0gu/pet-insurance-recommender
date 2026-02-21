from langchain_core.documents import Document

chunk = Document(
    page_content=('- 는 서면(등기우편 등)으로 다시 알려드립니다.\n'
 '- \uf000 제4항에도 불구하고 손해가 제1항 제1호 및 제2호의 사실로 생긴 것이 아님을 계\n'
 '- 약을 해지할 수 없습니다.\n'
 '- 1. 회사가 최초 계약 체결 당시에 그 사실을 알았거나 과실로 알지 못하였을 때\n'
 '- 2. 회사가 그 사실을 안 날부터 1개월 이상 지났거나 또는 제1회 보험료 등을 받\n'
 '- 은 때부터 보험금 지급사유가 발생하지 않고 2년이 지났을 때\n'
 '- 3. 최초 계약을 체결한 날부터 3년이 지났을 때'),
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
