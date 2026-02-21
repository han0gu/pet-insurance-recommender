from langchain_core.documents import Document

chunk = Document(
    page_content=('- 약한 사람을 말합니다.\n'
 '64 KB 금쪽같은 펫보험(강아지)(무배당)(26.01)제22조(계약내용의 변경 등)\uf000등 기타 계약의 내용\n'
 '\uf000 계약자는 보험수익자를 변경할 수 있으며 이 경우에는 회사의 승낙이 필요하지 않\n'
 '습니다. 다만, 변경된 보험수익자가 회사에 권리를 대항하기 위해서 계약자는 보험- 계약자는 회사의 승낙을 얻어 다음의 사항을 변경할 수 '
 '있습니다. 이 경우 승낙을\n'
 '- 서면 등으로 알리거나 보험증권의 뒷면에 기재하여 드립니다.\n'
 '- 1. 보험종목\n'
 '- 2. 보험기간\n'
 '- 3. 보험료 납입방법 및 납입기간'),
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
