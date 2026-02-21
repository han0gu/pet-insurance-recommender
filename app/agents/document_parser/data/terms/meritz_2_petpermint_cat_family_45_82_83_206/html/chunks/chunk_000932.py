from langchain_core.documents import Document

chunk = Document(
    page_content=('청력을 완전히 잃었을 때</td><td>25</td></tr><tr><td>4) 한 귀의 청력에 심한 장해를 남긴 '
 '때</td><td>15</td></tr><tr><td>5) 한 귀의 청력에 약간의 장해를 남긴 '
 '때</td><td>5</td></tr><tr><td>6) 한 귀의 귓바퀴의 대부분이 결손된 때</td><td>1 '
 '0</td></tr><tr><td>7) 평형기능에 장해를 남긴 때</td><td>10</td></tr></tbody></table><h1 '
 "id='17' style='font-size:16px'>나"),
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
