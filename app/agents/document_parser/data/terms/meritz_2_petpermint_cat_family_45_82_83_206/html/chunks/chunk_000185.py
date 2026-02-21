from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:14px'>71</footer><p id='51' data-category='paragraph' "
 "style='font-size:20px'>제28조(보험료의 자동대출납입)</p><br><p id='52' "
 "data-category='paragraph' style='font-size:16px'>\uf000 계약자는 제29조(보험료의 납입이 "
 '연체되는 경우 납입최<br>고(독촉)와 계약의 해지)에 따른 보험료의 납입최고(독촉)<br>기간이 지나기 전까지 회사가 정한 방법에 따라 '
 '보험료의<br>자동대출납입을 신청할 수'),
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
