from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만,【별표2(장<br>해분류표)】의 각 신체부위별 판정기준에 별도로 정한 경우<br>에는 그 기준에 따릅니다.<br>\uf000 '
 "다른 상해로 인하여 후유장해가 2회 이상 발생하였을 경</p><footer id='34' "
 "style='font-size:14px'>50</footer><p id='35' data-category='paragraph' "
 "style='font-size:16px'>우에는 그 때마다 이에 해당하는 후유장해지급률을 결정합<br>니다"),
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
