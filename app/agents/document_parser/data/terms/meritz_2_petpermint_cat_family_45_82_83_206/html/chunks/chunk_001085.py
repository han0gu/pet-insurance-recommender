from langchain_core.documents import Document

chunk = Document(
    page_content=('남긴때</td><td>100</td></tr><tr><td>3) 정신행동에 심한 장해를 남긴 '
 '때</td><td>75</td></tr><tr><td>4) 정신행동에 뚜렷한 장해를 남긴 '
 '때</td><td>50</td></tr><tr><td>5) 정신행동에 약간의 장해를 남긴 '
 '때</td><td>25</td></tr><tr><td>6) 정신행동에 경미한 장해를 남긴 '
 '때</td><td>10</td></tr><tr><td>7) 극심한 치매 : CDR 척도 '
 '5점</td><td>100</td></tr><tr><td>8) 심한 치매 : CDR'),
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
