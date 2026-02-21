from langchain_core.documents import Document

chunk = Document(
    page_content=('(양성 또는 악성이 불확실 한)</td></tr><tr><td>AFA008</td><td>흑색종 '
 '(양성)</td></tr><tr><td>AFB008</td><td>흑색종 '
 '(악성)</td></tr><tr><td>AFC008</td><td>흑색종 (양성 또는 악성이 '
 '불확실한)</td></tr><tr><td>AFB009</td><td>피부'),
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
