from langchain_core.documents import Document

chunk = Document(
    page_content=('- 청약일 현재 유지중이거나, 계약 청약일 전 6개월 이내에 계약자 및 피보험자의\n'
 '- 요구 또는 보험료 납입 연체로 해지된 경우 유사계약에서 정한 부담보 기간 종료\n'
 '- 일 이내에서 계약의 부담보 기간을 적용하고, 유사계약에서 정한 질병과 동일하거\n'
 '- 나 축소된 범위로 계약의 부담보 설정 범위를 정합니다. 또한 유사계약이 다수인\n'
 '- 경우 해당 반려동물에게 가장 유리한 계약조건을 적용합니다.\n'
 '- 단, 유사계약 청약일 이후 제1항에서 정한 질병과 관련한 새로운 위험(재진단·치'),
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
