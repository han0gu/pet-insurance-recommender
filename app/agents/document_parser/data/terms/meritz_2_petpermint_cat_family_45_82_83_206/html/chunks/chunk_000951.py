from langchain_core.documents import Document

chunk = Document(
    page_content=('약간의 장 해를 남긴 때</td><td>5</td></tr><tr><td>8) 치아에 14개 이상의 결손이 생긴 '
 '때</td><td>20</td></tr><tr><td>9) 치아에 7개 이상의 결손이 생긴 '
 '때</td><td>10</td></tr><tr><td>10) 치아에 5개 이상의 결손이 생긴 '
 "때</td><td>5</td></tr></tbody></table><footer id='38' "
 "style='font-size:14px'>181</footer><h1 id='39' style='font-size:20px'>나"),
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
