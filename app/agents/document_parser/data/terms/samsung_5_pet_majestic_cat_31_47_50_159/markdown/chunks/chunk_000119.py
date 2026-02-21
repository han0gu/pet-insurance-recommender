from langchain_core.documents import Document

chunk = Document(
    page_content=('- 이율로 계산한 금액을 더하여 납입하여야 합니다. 다만, 금리연동형보험은 각 상품별\n'
 '- 사업방법서에서 별도로 정한 이율로 계산합니다.\n'
 '- ② 제1항에 따라 해지계약을 부활(효력회복)하는 경우에는 제16조(계약 전 알릴 의무),\n'
 '- 제18조(알릴 의무 위반의 효과), 제19조(사기에 의한 계약), 제20조(보험계약의 성립)\n'
 '- 및 제27조(제1회 보험료 및 회사의 보장개시)를 준용합니다. 이때 회사는 해지 전 발\n'
 '- 생한 보험금 지급사유를 이유로 부활(효력회복)을 거절하지 않습니다.'),
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
