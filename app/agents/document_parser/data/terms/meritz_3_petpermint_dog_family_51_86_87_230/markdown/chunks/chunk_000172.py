from langchain_core.documents import Document

chunk = Document(
    page_content=('[위험변경에 따른 계약변경 절차]\n'
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
 '계약변경 완료\uf000 회사는 제2항에 따라 계약내용을 변경할 때 위험이 감소\n'
 '된 경우에는 보험료를 감액하고, 이후 기간 보장을 위한 재\n'
 '원인 계약자적립액 등의 차이로 인하여 발생한 정산금액(이\n'
 '하 「정산금액」이라 합니다)을 환급하여 드립니다. 한편\n'
 '위험이 증가된 경우에는 보험료의 증액 및 정산금액의 추가'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
