from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:14px'>- 15 -</footer><h1 id='53' "
 "style='font-size:14px'>제27조(보험료의 납입이 연체되는 경우 납입최고(독촉)와 계약의 해지)</h1><br><p "
 "id='54' data-category='paragraph' style='font-size:14px'>① 계약자가 제2회 이후의 보험료를 "
 '납입기일까지 납입하지 않아 보험료 납입이 연체 중<br>인 경우에 회사는 14일(보험기간이 1년 미만인 경우에는 7일) 이상의 기간을'),
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
