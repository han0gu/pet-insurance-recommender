from langchain_core.documents import Document

chunk = Document(
    page_content=('- 에 회사는 보험금을 지급하지 않으며, 계약 전 알릴 의무 위반 사실(계약해지 등의 원\n'
 '- 인이 되는 위반사실을 구체적으로 명시)뿐만 아니라 계약 전 알릴 의무사항이 중요한\n'
 '- 사항에 해당되는 사유를 "반대증거가 있는 경우 이의를 제기할 수 있습니다"라는 문구\n'
 '- 와 함께 계약자에게 서면 또는 전자문서 등으로 알려 드립니다. 회사가 전자문서로 안\n'
 '- 내하고자 할 경우에는 계약자에게 서면 또는 「전자서명법」 제2조 제2호에 따른 전\n'
 '- 자서명으로 동의를 얻어 수신확인을 조건으로 전자문서를 송신하여야 합니다. 계약자'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
