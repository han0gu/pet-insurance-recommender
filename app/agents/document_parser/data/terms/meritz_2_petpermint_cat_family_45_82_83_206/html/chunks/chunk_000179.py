from langchain_core.documents import Document

chunk = Document(
    page_content=('. 자동이체<br>또는 신용카드로 납입하는 경우에는 자동이체신청 또는 신<br>용카드매출승인에 필요한 정보를 제공한 때를 제1회 '
 '보험료<br>를 받은 때로 하며, 계약자의 책임 있는 사유로 자동이체<br>또는 매출승인이 불가능한 경우에는 보험료가 납입되지 '
 '않<br>은 것으로 봅니다.<br>\uf000 회사가 청약과 함께 제1회 보험료를 받고 청약을 승낙하<br>기 전에 보험금 지급사유가 '
 "발생하였을 때에도 보장개시일<br>부터 이 약관이 정하는 바에 따라 보장을 합니다.</p><br><h1 id='42'"),
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
