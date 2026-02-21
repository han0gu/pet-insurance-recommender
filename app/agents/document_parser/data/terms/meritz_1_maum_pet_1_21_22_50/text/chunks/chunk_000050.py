from langchain_core.documents import Document

chunk = Document(
    page_content=('경 등)에 따라 계약내용을 변경할 수 있습니다.![image](/image/placeholder)\n'
 '[위험변경에 따른 계약변경 절차]\n'
 '위험변경사항 통지\n'
 '(우편, 전화, 방문 등)\n'
 '↓\n'
 '계약자,피보험자의 계약변경사항 확인 후 청약\n'
 '↓\n'
 '계약변경사항 인수 심사\n'
 '↓\n'
 '정산금액 처리\n'
 '(환급 또는 추가납입)\n'
 '↓\n'
 '계약변경 완료③ 회사는 제2항에 따라 계약내용을 변경할 때 위험이 감소된 경우에는 납입보험료를 감액\n'
 '하고, 이후 기간 보장을 위한 재원인 해약환급금 등의 차이로 인하여 발생한 정산금액(이'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
