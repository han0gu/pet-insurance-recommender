from langchain_core.documents import Document

chunk = Document(
    page_content=('(원인 불명)</td></tr><tr><td>HAA007</td><td>확장성 '
 '심근병증</td></tr><tr><td>HAA008</td><td>비대성 '
 '심근병증</td></tr><tr><td>HAA009</td><td>제한성 '
 '심근병증</td></tr><tr><td>HAA010</td><td>일시적 '
 '심근비대증</td></tr><tr><td>HAA011</td><td>기타 '
 '심근증</td></tr><tr><td>HAA012</td><td>대동맥 협착증 · '
 'AS</td></tr><tr><td>HAA013</td><td>폐동맥 협착 ·'),
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
