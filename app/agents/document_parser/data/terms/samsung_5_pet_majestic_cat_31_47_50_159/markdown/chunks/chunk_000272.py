from langchain_core.documents import Document

chunk = Document(
    page_content=('기간이 종료된 이후 계약이 해지될 경우 표준형 상품 해약환급금의 50%에 해당하\n'
 '는 금액을 지급합니다.<유의사항># [해약환급금 미지급형Ⅱ의 해약환급금 관련]- • 보험료 납입기간 중 계약이 해지될 경우 해약환급금이 '
 '없고, 보험료 납입기간이 완료된 이후\n'
 '- 계약이 해지될 경우 표준형 상품 대비 적은 해약환급금을 지급하는 대신 표준형 상품보다 낮은\n'
 '- 보험료로 가입할 수 있도록 한 상품입니다.\n'
 '- • 해약환급금을 계산할 때 기준이 되는 표준형 상품의 해약환급금은 “보험료 및 해약환급금'),
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
