from langchain_core.documents import Document

chunk = Document(
    page_content=('- 는 보험금을 지급하지 않으며, 계약 전 알릴 의무 위반 사실(계약해지 등의 원인이 되\n'
 '- 는 위반사실을 구체적으로 명시) 뿐만 아니라 계약 전 알릴 의무사항이 중요한 사항에\n'
 '- 해당되는 사유를 "반대증거가 있는 경우 이의를 제기할 수 있습니다"라는 문구와 함께\n'
 '- 계약자에게 서면 또는 전자문서 등으로 알려 드립니다. 회사가 전자문서로 안내하고\n'
 '- 자 할 경우에는 계약자에게 서면 또는 「전자서명법」 제2조 제2호에 따른 전자서명\n'
 '- 으로 동의를 얻어 수신확인을 조건으로 전자문서를 송신하여야 합니다. 계약자의 전'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
