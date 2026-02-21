from langchain_core.documents import Document

chunk = Document(
    page_content=('우. 다만, 회사가 나이의 착오를 발견하였을 때 이미 계약나이에 도달한 경우에는\n'
 '유효한 계약으로 보나, 제2호의 만 15세 미만자에 관한 예외가 인정되는 것은 아\n'
 '닙니다.# 제21조 (계약내용의 변경 등)① 계약자는 회사의 승낙을 얻어 다음의 사항을 변경할 수 있습니다. 이 경우 승낙을 서\n'
 '면 등으로 알리거나 보험증권의 뒷면에 기재하여 드립니다.- 1. 보험종목\n'
 '- 2. 보험기간\n'
 '- 3. 보험료 납입주기, 납입방법 및 납입기간\n'
 '- 4. 계약자, 피보험자\n'
 '- 5. 보험가입금액 등 기타 계약의 내용'),
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
