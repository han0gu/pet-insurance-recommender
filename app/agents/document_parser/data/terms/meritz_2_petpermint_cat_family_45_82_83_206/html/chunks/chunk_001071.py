from langchain_core.documents import Document

chunk = Document(
    page_content=(". 장해의 분류</h1><br><table id='15' style='font-size:16px'><thead><tr><td>장해의 "
 '분류</td><td>지급률</td></tr></thead><tbody><tr><td>1) 심장 기능을 잃었을 '
 '때</td><td>100</td></tr><tr><td>2) 흉복부장기 또는 비뇨생식기 기능을 잃었을 '
 '때</td><td>75</td></tr><tr><td>3) 흉복부장기 또는 비뇨생식기 기능에 심한 장 해를 남긴 '
 '때</td><td>50</td></tr><tr><td>4) 흉복부장기 또는'),
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
