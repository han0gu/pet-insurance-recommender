from langchain_core.documents import Document

chunk = Document(
    page_content=('- \uf000 제1항에 따라 계약을 해지하였을 때에는 제34조(해약환급금) 제1항에 따른 해약환\n'
 '- 급금을 계약자에게 지급합니다.\n'
 '- \uf000 제1항 제1호에 의한 계약의 해지가 보험금 지급사유 발생 후에 이루어진 경우에 회\n'
 '- 사는 보험금을 지급하지 않으며, 계약 전 알릴 의무 위반사실(계약해지 등의 원인\n'
 '- 이 되는 위반사실을 구체적으로 명시)뿐만 아니라 계약 전 알릴 의무사항이 중요\n'
 '- 한 사항에 해당되는 사유를 "반대증거가 있는 경우 이의를 제기할 수 있습니다"라\n'
 '- 는 문구와 함께 계약자에게 서면 또는 전자문서 등으로 알려 드립니다.'),
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
