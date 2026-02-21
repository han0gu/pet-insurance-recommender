from langchain_core.documents import Document

chunk = Document(
    page_content=('한 눈의 교정시력이 0.06 이하로 된 때</td><td>25</td></tr><tr><td>5) 한 눈의 교정시력이 0.1 이하로 된 '
 '때</td><td>15</td></tr><tr><td>6) 한 눈의 교정시력이 0.2 이하로 된 '
 '때</td><td>5</td></tr><tr><td>7) 한눈의 안구(눈동자)에 뚜렷한 운동장해나 뚜렷한 조절기능장해를 남긴 '
 '때</td><td>10</td></tr><tr><td>8) 한 눈에 뚜렷한 시야장해를 남긴 '
 '때</td><td>5</td></tr><tr><td>9) 한눈의 눈꺼풀에 뚜렷한 결손을'),
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
