from langchain_core.documents import Document

chunk = Document(
    page_content=('- 회사는 보험금을 지급하지 않으며, 보험료 납입면제사유가 발생한 경우 보험료 납입\n'
 '- 을 면제하지 않습니다. 또한, 계약 전 알릴 의무 위반 사실(특별약관 해지 등의 원인이\n'
 '- 되는 위반사실을 구체적으로 명시)뿐만 아니라 계약 전 알릴 의무사항이 중요한 사항\n'
 '- 에 해당되는 사유를 "반대증거가 있는 경우 이의를 제기할 수 있습니다"라는 문구와\n'
 '- 함께 계약자에게 서면 또는 전자문서 등으로 알려드립니다. 회사가 전자문서로 안내\n'
 '- 하고자 할 경우에는 계약자에게 서면 또는 「전자서명법」 제2조 제2호에 따른 전자'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
