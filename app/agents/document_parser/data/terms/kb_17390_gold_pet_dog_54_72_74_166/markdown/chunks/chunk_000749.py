from langchain_core.documents import Document

chunk = Document(
    page_content=('- \uf000 이 특별약관은 보험계약(보통약관 및 다른 특별약관이 부가된 경우에는 그 특별약\n'
 '- 관도 포함합니다, 이하 "보험계약"이라 합니다)을 체결할 때 계약자의 청약과 회\n'
 '- 사의 승낙으로 보험계약에 부가하여 이루어집니다.\n'
 '- \uf000 이 특별약관의 효력발생일은 보통약관 제1절 일반조항 제25조(제1회 보험료 및 회\n'
 '- 사의 보장개시)에서 정한 보장개시일과 동일합니다.\n'
 '- \uf000 보험계약이 해지, 기타사유에 의하여 효력을 가지지 않게 된 경우에는 이 특별약\n'
 '- 관도 더 이상 효력을 가지지 않습니다.'),
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
