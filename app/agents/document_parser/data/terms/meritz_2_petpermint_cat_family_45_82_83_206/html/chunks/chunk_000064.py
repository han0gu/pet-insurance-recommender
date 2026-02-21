from langchain_core.documents import Document

chunk = Document(
    page_content=('대하여 나누어 지급받거나 일시에 지급받는 방법으<br>로 변경할 수 있습니다.<br>\uf000 회사는 제1항에 따라 일시에 지급할 '
 '금액을 나누어 지급<br>하는 경우에는 나중에 지급할 금액에 대하여 평균공시이율<br>을 연단위 복리로 계산한 금액을 더하며, 나누어 '
 '지급할 금<br>액을 일시에 지급하는 경우에는 평균공시이율을 연단위 복<br>리로 할인한 금액을 지급합니다.</p><br><h1 '
 "id='89' style='font-size:16px'>【보험금 지급 예시】</h1><br><p id='90'"),
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
