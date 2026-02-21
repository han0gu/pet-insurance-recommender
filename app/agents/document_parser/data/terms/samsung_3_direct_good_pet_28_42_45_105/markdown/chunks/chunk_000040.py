from langchain_core.documents import Document

chunk = Document(
    page_content=('③ 회사는 제2항에 따라 계약내용을 변경할 때 위험이 감소된 경우에는 보험료를 감액하\n'
 '고, 이후 기간 보장을 위한 재원인 계약자적립액 등의 차이로 인하여 발생한 정산금액\n'
 '(이하 「정산금액」 이라 합니다)을 환급하여 드립니다. 한편 위험이 증가된 경우에는\n'
 '보험료의 증액 및 정산금액의 추가납입을 요구할 수 있으며, 계약자는 일시납 또는 잔\n'
 '여 보험료 납입기간과 5년 중 큰 기간(단, 잔여 보험기간을 초과할 수 없음) 동안의- 32 -④ 제1항의 통지에 따라 위험의 증가로 '
 '보험료를 더 내야 할 경우 회사가 청구한 추가보'),
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
