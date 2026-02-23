from langchain_core.documents import Document

chunk = Document(
    page_content=('제23조(타인을 위한 계약)\n'
 '\uf000 계약자는 타인을 위한 계약을 체결하는 경우에 그 타인의 위임이 없는 때에는 반\n'
 '드시 이를 회사에 알려야 하며, 이를 알리지 않았을 때에는 그 타인은 이 계약이\n'
 '체결된 사실을 알지 못하였다는 사유로 회사에 이의를 제기할 수 없습니다.\n'
 '\uf000 타인을 위한 계약에서 보험사고가 발생한 경우에 계약자가 그 타인에게 보험사고 상\n'
 '의 발생으로 생긴 손해를 배상한 때에는 계약자는 그 타인의 권리를 해하지 않는 해- \n'
 '| 범위 안에서 회사에 | 보험금의 지급을 청구할 수 있습니다. |\n'
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
