from langchain_core.documents import Document

chunk = Document(
    page_content=('- 위하여 필요한 서류를 효력상실 또는 해지 즉시 회사에 제출하여야 합니다.\n'
 '- 2. 회사는 보험기간중이나 보험기간 만료후 보험료를 산출하기 위하여 필요하다고 인정될 경우에\n'
 '- 는 계약자의 서류를 열람할 수 있습니다.\n'
 '- 3. 회사는 보험기간 만료와 동시에 제1호에의한 피보험자수에 따라 산출된 확정보험료와 기납입한\n'
 '- 보험료를 비교하여 그 차액을 정산합니다.\n'
 '- 4. 제1호에도 불구하고, 계약자와 협의를 통해 피보험자수에 관한 서류 제출 주기를 변경할 수 있\n'
 '- 습니다.'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
