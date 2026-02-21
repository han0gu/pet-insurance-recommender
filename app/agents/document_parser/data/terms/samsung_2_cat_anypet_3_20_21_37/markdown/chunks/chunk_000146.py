from langchain_core.documents import Document

chunk = Document(
    page_content=('- 사람 또는 「장애인복지법」 에 따른 장애인등록증을 발급받은 사람에 대해서는 해당 증명서·장애\n'
 '- 인등록증의 사본이나 그 밖의 장애 사실을 증명하는 서류를 제출하는 경우에는 제 1항의 장애인증\n'
 '- 명서는 제출하지 않을 수 있습니다.\n'
 '- ③ 장애인으로서 그 장애기간이 기재된 장애인증명서를 제1항 따라 회사에 제출한 때에는 그 장애기\n'
 '- 간 동안은 이를 다시 제출하지 않을 수 있습니다.\n'
 '- ④ 제1항에 따라 제출한 장애인증명서의 장애기간이 변경되는 경우 계약자는 이를 회사에 알리고 변'),
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
