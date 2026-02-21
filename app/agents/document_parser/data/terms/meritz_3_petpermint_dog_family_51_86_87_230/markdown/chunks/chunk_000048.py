from langchain_core.documents import Document

chunk = Document(
    page_content=('- 합니다.)\n'
 '\uf000 회사는 제1항의 통지로 인하여 위험의 변동이 발생한 경\n'
 '우에는 보통약관 제23조(계약내용의 변경 등)에 따라 계약\n'
 '내용을 변경할 수 있습니다.![image](/image/placeholder)\n'
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
 '계약변경 완료\uf000 회사는 제2항에 따라 계약내용을 변경할 때 위험이 감소'),
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
