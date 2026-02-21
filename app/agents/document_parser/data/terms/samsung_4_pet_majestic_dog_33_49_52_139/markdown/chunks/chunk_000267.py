from langchain_core.documents import Document

chunk = Document(
    page_content=('납입이 완료되고 보험료 납입기간이 종료된 이후 계약이 해지될 경우 표준형 상품\n'
 '해약환급금의 100%에 해당하는 금액을 지급합니다.<유의사항># [해약환급금 일부지급형의 해약환급금 관련]- • 보험료 납입기간 중 '
 '계약이 해지될 경우 표준형 상품 대비 적은 해약환급금을 지급하는 대신\n'
 '- 표준형 상품보다 낮은 보험료로 가입할 수 있도록 한 상품입니다.\n'
 '- • 해약환급금을 계산할 때 기준이 되는 표준형 상품의 해약환급금은 “보험료 및 해약환급금\n'
 '- 산출방법서”에 따라 계산한 금액으로 해지율을 적용하지 않고 계산합니다.'),
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
