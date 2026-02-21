from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만,<br>타인을 위한 계약의 경우에는 계약자는 그 타인의 동의를<br>얻거나 보험증권을 소지한 경우에 한하여 계약을 해지할 '
 "수<br>있습니다.</p><h1 id='58' style='font-size:20px'>제21조(중대사유로 인한 "
 "해지)</h1><br><p id='59' data-category='paragraph' "
 "style='font-size:16px'>\uf000 회사는 아래와 같은 사실이 있을 경우에는 안 날부터 1<br>개월 이내에 계약을 "
 "해지할 수 있습니다.</p><br><p id='60'"),
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
