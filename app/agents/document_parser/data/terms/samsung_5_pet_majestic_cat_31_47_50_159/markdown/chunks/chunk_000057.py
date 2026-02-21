from langchain_core.documents import Document

chunk = Document(
    page_content=('- 고, 이후 기간 보장을 위한 재원인 계약자적립액 등의 차이로 인하여 발생한 정산금액\n'
 '- (이하 「정산금액」이라 합니다)을 환급하여 드립니다. 한편 위험이 증가된 경우에는\n'
 '- 보험료의 증액 및 정산금액의 추가납입을 요구할 수 있으며, 계약자는 일시납 또는 잔\n'
 '- 여 보험료 납입기간과 5년 중 큰 기간(단, 잔여 보험기간을 초과할 수 없음) 동안의\n'
 '- 분납 중 선택하여 정산금액을 납입하여야 합니다. 다만, 보험료 갱신형 계약 등 회사\n'
 '- 가 정하는 기준에 따라 일부 보험계약의 경우 분납이 제한될 수 있습니다.'),
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
