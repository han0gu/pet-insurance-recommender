from langchain_core.documents import Document

chunk = Document(
    page_content=('었을 경우. 다만, 회사가 나이의 착오를 발견하였을 때 이미 계약나이에 도달한 경우에\n'
 '는 유효한 계약으로 봅니다.# 제23조(계약내용의 변경 등)① 계약자는 회사의 승낙을 얻어 다음의 사항을 변경할 수 있습니다. 이 경우 '
 '승낙을 서면\n'
 '등으로 알리거나 보험증권의 뒷면에 기재하여 드립니다.- 1. 보험종목\n'
 '- 2. 보험기간\n'
 '- 3. 보험료 납입주기, 납입방법 및 납입기간\n'
 '- 4. 계약자, 피보험자 중 일부\n'
 '- 5. 보험가입금액, 보험료, 배상책임의 경우 보상한도액 등 기타 계약의 내용'),
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
