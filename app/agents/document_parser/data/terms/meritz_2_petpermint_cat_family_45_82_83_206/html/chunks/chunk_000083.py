from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:20px'>【사례】</h1><br><p id='20' data-category='paragraph' "
 "style='font-size:20px'>계약 청약을 하면서 보험설계사에게 고혈압이 있다고만<br>얘기하였을 뿐, 청약서의 계약 전 알릴 "
 '사항에 아무런<br>기재도 하지 않았을 경우에는 보험설계사에게만 고혈압<br>병력을 얘기하였다고 하더라도 회사는 계약 전 알릴 '
 "의<br>무 위반을 이유로 계약을 해지하고 보험금을 지급하지<br>않을 수 있습니다.</p><h1 id='21'"),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
