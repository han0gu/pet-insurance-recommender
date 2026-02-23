from langchain_core.documents import Document

chunk = Document(
    page_content=('중요한 사항을 고지하지 아니하거나 부실의 고지를 한 때에는 보험자는 그 사실\n'
 '을 안 날로부터 1월내에, 계약을 체결한 날로부터 3년내에 한하여 계약을 해지할\n'
 '제\n'
 '수 있다. 그러나 보험자가 계약당시에 그 사실을 알았거나 중대한 과실로 인하여\n'
 '도\n'
 '알지 못한 때에는 그러하지 아니하다.\n'
 '성\n'
 '∙ 상법 제651조의2(서면에 의한 질문의 효력)\n'
 '특'),
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
