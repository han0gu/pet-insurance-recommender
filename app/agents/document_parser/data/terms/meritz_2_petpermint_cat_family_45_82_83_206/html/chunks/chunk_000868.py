from langchain_core.documents import Document

chunk = Document(
    page_content=('결석</td></tr><tr><td>OAA006 OAA007</td><td>방광염 방광 '
 '결석</td></tr><tr><td>OAA008</td><td>요도 폐색</td></tr><tr><td>OAA009</td><td>요로 '
 '결석증</td></tr><tr><td>OAA010</td><td>신경성 배뇨 '
 '이상</td></tr><tr><td>OAA011</td><td>고양이 특발성 '
 '방광염(FIC)</td></tr><tr><td>OAA012</td><td>고양이 하부 비뇨기계'),
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
