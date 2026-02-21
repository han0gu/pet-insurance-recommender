from langchain_core.documents import Document

chunk = Document(
    page_content=('- 2. 보험기간\n'
 '- 3. 보험료 납입주기, 납입방법 및 납입기간\n'
 '- 4. 계약자, 피보험자\n'
 '- 5. 보험가입금액, 보험료 등 기타 계약의 내용\n'
 '- ② 회사는 계약자가 제1회 보험료 등을 납입한 때부터 1년 이상 지난 유효한 계약으로서 그 보험종목\n'
 '- 의 변경을 요청할 때에는 회사의 사업방법서에서 정하는 방법에 따라 이를 변경하여 드립니다.\n'
 '- ③ 회사는 계약자가 제1항 제5호의 규정에 의하여 보험가입금액을 감액하고자 할 때에는 그 감액된 부'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
