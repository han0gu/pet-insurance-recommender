from langchain_core.documents import Document

chunk = Document(
    page_content=('- 금을 계약자에게 지급합니다.\n'
 '- ④ 제1항 제1호에 의한 계약의 해지가 보험금 지급사유 발생 후에 이루어진 경우에 회사\n'
 '- 는 보험금을 지급하지 않으며, 계약 전 알릴 의무 위반 사실(계약해지 등의 원인이 되\n'
 '- 는 위반사실을 구체적으로 명시) 뿐만 아니라 계약 전 알릴 의무사항이 중요한 사항에\n'
 '- 해당되는 사유를 "반대증거가 있는 경우 이의를 제기할 수 있습니다"라는 문구와 함께\n'
 '- 계약자에게 서면 또는 전자문서 등으로 알려드립니다. 회사가 전자문서로 안내하고자'),
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
