from langchain_core.documents import Document

chunk = Document(
    page_content=('- (독촉)기간 내에 연체보험료를 납입하여야 한다는 내용\n'
 '- 2. 납입최고(독촉)기간이 끝나는 날까지 보험료를 납입하지 않을 경우 납입최고(독촉)기\n'
 '- 간이 끝나는 날의 다음날에 계약이 해지된다는 내용\n'
 '- ② 회사가 제1항에 따른 납입최고(독촉) 등을 전자문서로 안내하고자 할 경우에는 계약자\n'
 '- 에게 서면, ⌜전자서명법⌟ 제2조 제2호에 따른 전자서명으로 동의를 얻어 수신확인을\n'
 '- 조건으로 전자문서를 송신하여야 하며, 계약자가 전자문서에 대하여 수신을 확인하기'),
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
