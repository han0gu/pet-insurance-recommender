from langchain_core.documents import Document

chunk = Document(
    page_content=('- 제기하지 않습니다.\n'
 '제44조(관할법원)이 계약에관한 소송 및 민사조정은 계약자의 주소지를 관할하는 법원으로 합니다. 다70 KB 금쪽같은 '
 '펫보험(강아지)(무배당)(26.01)# 만, 회사와 계약자가# 합의하여 관할법원을 달리 정할 수 있습니다.제45조(소멸시효)| 적립액 '
 '반환청구권은 3년간 | 행사하지 않으면 소멸시효가 완성됩니다. |\n'
 '| --- | --- |'),
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
