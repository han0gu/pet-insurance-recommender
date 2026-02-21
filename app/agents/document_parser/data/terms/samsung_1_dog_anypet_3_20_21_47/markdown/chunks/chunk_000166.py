from langchain_core.documents import Document

chunk = Document(
    page_content=('정해진 보험요율을 적용하여 산출한 보험료(이하 「예치보험료」 라 합니다)를 회사에 납입하여야 합니다.# 제6조(보험료의 정산방법)- ① '
 '계약자는 계약이 효력상실 또는 해지된 경우에는 효력상실 또는 해지일까지의 보험료를 확정하기\n'
 '- 위하여 필요한 서류를 효력상실 또는 해지 즉시 회사에 제출해야 합니다.\n'
 '- ② 회사는 보험기간 중이나 보험기간 만료 후 보험료를 산출하기 위하여 필요하다고 인정될 경우에는\n'
 '- 계약자의 서류를 열람할 수 있습니다.'),
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
