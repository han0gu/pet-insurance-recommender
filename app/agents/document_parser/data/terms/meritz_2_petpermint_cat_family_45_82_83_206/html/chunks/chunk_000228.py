from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:20px'>보험계약대출의 원금과 이자를 차감할 수 있습니다.</p><br><p id='11' "
 "data-category='paragraph' style='font-size:16px'>\uf000 제2항의 규정에도 불구하고 회사는 "
 '제29조(보험료의 납<br>입이 연체되는 경우 납입최고(독촉)와 계약의 해지)에 따라<br>계약이 해지되는 때에는 즉시 해약환급금에서 '
 '보험계약대출<br>의 원금과 이자를 차감합니다.<br>\uf000 회사는 보험수익자에게 보험계약대출 사실을 통지할 '
 "수<br>있습니다.</p><h1 id='12'"),
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
