from langchain_core.documents import Document

chunk = Document(
    page_content=(". 장해의 분류</h1><br><table id='73' style='font-size:20px'><thead><tr><td>장해의 "
 '분류</td><td>지급률</td></tr></thead><tbody><tr><td>1) 한손의 5개 손가락을 모두 잃었을 '
 '때</td><td>55</td></tr><tr><td>2) 한손의 첫째 손가락을 잃었을 '
 '때</td><td>15</td></tr><tr><td>3) 한손의 첫째 손가락 이외의 손가락을 잃었을 때 (손가락 '
 '하나마다)</td><td>10</td></tr><tr><td>4) 한손의'),
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
