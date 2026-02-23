from langchain_core.documents import Document

chunk = Document(
    page_content=('을 때</td><td>30</td></tr><tr><td>4) 한다리의 3대관절중 관절 하나의 기능에 심한 장해 를 남긴 '
 '때</td><td>20</td></tr><tr><td>5) 한다리의 3대관절중 관절 하나의 기능에 뚜렷한 장해 를 남긴 '
 '때</td><td>10</td></tr><tr><td>6) 한다리의 3대관절중 관절 하나의 기능에 약간의 장해 를 남긴 '
 '때</td><td>5</td></tr><tr><td>7) 한다리에 가관절이 남아 뚜렷한 장해를 남긴 '
 '때</td><td>20</td></tr><tr><td>8) 한다리에'),
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
