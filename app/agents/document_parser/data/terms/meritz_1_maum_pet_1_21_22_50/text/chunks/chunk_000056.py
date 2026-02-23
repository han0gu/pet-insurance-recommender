from langchain_core.documents import Document

chunk = Document(
    page_content=('진단서 사본 등)에 따라 승낙한 경우에 건강진단서 사본 등에 명기되어 있는 사항으\n'
 '로 보험금 지급사유가 발생하였을 때(계약자 또는 피보험자가 회사에 제출한 기초자\n'
 '료의 내용 중 중요사항을 고의로 사실과 다르게 작성한 때에는 계약을 해지할 수\n'
 '있습니다)\n'
 '5. 보험설계사 등이 계약자 또는 피보험자에게 고지할 기회를 주지 않았거나 계약자 또\n'
 '는 피보험자가 사실대로 고지하는 것을 방해한 경우, 계약자 또는 피보험자에게 사\n'
 '실대로 고지하지 않게 하였거나 부실한 고지를 권유했을 때. 다만, 보험설계사 등의'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
