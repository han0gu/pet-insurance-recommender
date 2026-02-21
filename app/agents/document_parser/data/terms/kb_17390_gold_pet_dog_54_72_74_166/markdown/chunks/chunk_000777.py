from langchain_core.documents import Document

chunk = Document(
    page_content=('무)를 적용합니다.- \n'
 '# 제3조(자동갱신 적용)- 134 -- \uf000 회사는 제2조(보장특약의 자동갱신)에 의하여 이 특별약관이 갱신되는 경우 최초\n'
 '- 계약시의 보험약관을 계속하여 적용합니다.\n'
 '- \uf000 회사는 갱신계약에 대하여 갱신일 현재의 보험요율에 관한 제도를 반영하여 계산\n'
 '- 된 보험료를 적용하며, 그 보험료는 나이의 증가, 보험료산출에 관한 기초율의 변\n'
 '- 동 등의 사유로 인하여 인상 또는 인하될 수 있습니다.\n'
 '- \uf000 회사는 제2조(보장특약의 자동갱신)에서 정한 갱신제한 사유 및 제2항의 갱신계약'),
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
