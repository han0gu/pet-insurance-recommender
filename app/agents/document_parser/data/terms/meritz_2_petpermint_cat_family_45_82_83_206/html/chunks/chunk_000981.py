from langchain_core.documents import Document

chunk = Document(
    page_content=('인한 뚜렷한 신경 장해</td><td>15</td></tr><tr><td>9) 추간판탈출증으로 인한 약간의 신경 '
 "장해</td><td>10</td></tr></tbody></table><h1 id='77' style='font-size:20px'>나"),
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
