from langchain_core.documents import Document

chunk = Document(
    page_content=('때</td><td>30</td></tr><tr><td>4) 한팔의 3대관절중 관절 하나의 기능에 심한 장해를 남 긴 '
 '때</td><td>20</td></tr><tr><td>5) 한팔의 3대관절중 관절 하나의 기능에 뚜렷한 '
 "장해</td><td>10</td></tr></tbody></table><footer id='26' "
 "style='font-size:14px'>190</footer><table id='27' "
 "style='font-size:16px'><thead><tr><td>장해의"),
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
