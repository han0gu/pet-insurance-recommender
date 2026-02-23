from langchain_core.documents import Document

chunk = Document(
    page_content=('미만인 계약은 상해 발생일 또는 질병의 진단확정<br>일부터 1년 이내)에 장해상태가 더 악화된 때에는 그<br>악화된 장해상태를 '
 "기준으로 장해지급률을 결정한다.</p><br><h1 id='37' style='font-size:18px'>2"),
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
