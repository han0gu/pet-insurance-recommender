from langchain_core.documents import Document

chunk = Document(
    page_content=('존 계약내용에 상응하는 새로운 보장내용으로 계약내용을\n'
 '변경할 수 있습니다.① 관련 법률의 개정 또는 폐지 등에 따라 약관에서 정한\n'
 '보험금 지급사유 판정기준이 변경되는 경우84- ② 관련 법률의 개정 또는 폐지 등에 따라 약관에서 정한\n'
 '- 보험금 지급사유의 판정이 불가능한 경우\n'
 '- ③ 관련 법률의 개정 또는 폐지 등에 따라 계약유지 필요\n'
 '- 가 없어지는 경우\n'
 '- ④ 기타 금융위원회 등의 명령이 있는 경우\n'
 '\uf000 회사는 제2항에 따라 계약이 변경되는 경우 계약내용 변\n'
 '경일의 15일 이전까지 서면(등기우편 등), 전화(음성녹취)'),
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
