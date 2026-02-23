from langchain_core.documents import Document

chunk = Document(
    page_content=('. 가산이율 적용시 금융위원회 또는 금융감독원이<br>정당한 사유로 인정하는 경우에는 해당 기간에<br>대하여 가산이율을 적용하지 '
 '않습니다.<br>6'),
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
