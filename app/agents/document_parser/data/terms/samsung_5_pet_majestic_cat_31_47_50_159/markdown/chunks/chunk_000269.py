from langchain_core.documents import Document

chunk = Document(
    page_content=('간이 종료된 이후 계약이 해지될 경우 표준형 상품의 해약환급률에 이 상품의 해\n'
 '지 시점까지 납입한 보험료를 곱한 금액을 지급합니다. 이 때, 표준형 상품의 해약\n'
 '환급률이란 표준형 상품의 해약환급금을 표준형 상품의 해지 시점까지 납입한 보\n'
 '험료로 나눈 비율을 말하며, 해지 시점까지 납입한 보험료란 보험가입금액의 감액\n'
 '등 변경사항을 반영하여 계산한 해지 시점의 보험료에 해지 시점까지의 납입회차'),
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
