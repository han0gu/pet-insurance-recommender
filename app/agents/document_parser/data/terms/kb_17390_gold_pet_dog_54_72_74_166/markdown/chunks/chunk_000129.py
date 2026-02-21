from langchain_core.documents import Document

chunk = Document(
    page_content=('- 않고 사망한 경우, 최초 지정된 보험수익자의 권리가 확정됩니다. 그러나 계약자\n'
 '- 가 사망한 경우 그 승계인이 보험수익자 변경에 관한 권리를 행사할 수 있다는 별\n'
 '- 도의 약정이 있는 경우에는 승계받은 계약자가 보험수익자를 변경할 수 있습니다.\n'
 '- \uf000 회사는 제1항에 따라 계약자를 변경한 경우, 변경된 계약자에게 보험증권 및 약관\n'
 '- 을 교부하고 변경된 계약자가 요청하는 경우 약관의 중요한 내용을 설명하여 드립\n'
 '- 니다.\n'
 '- \uf000 제1항에 따라 위험이 증가하거나 감소되는 경우 납입보험료가 변경될 수 있으며,'),
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
