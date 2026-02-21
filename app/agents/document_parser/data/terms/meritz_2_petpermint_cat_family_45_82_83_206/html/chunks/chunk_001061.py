from langchain_core.documents import Document

chunk = Document(
    page_content=('한발의 첫째발가락 이외의 발가락을 잃었을 때 (발가락 하나마다)</td><td>5</td></tr><tr><td>5) 한발의 5개발가락 '
 '모두의 발가락뼈 일부를 잃었을 때 또는 뚜렷한 장해를 남긴 때</td><td>20</td></tr><tr><td>6) 한발의 첫째발가락의 '
 '발가락뼈 일부를 잃었을 때 또는 뚜렷한 장해를 남긴 때</td><td>8</td></tr><tr><td>7) 한발의 첫째발가락 이외의 '
 '발가락의 발가락 뼈 일부를 잃었을 때 또는 뚜렷한 장해를 남 긴 때(발가락'),
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
