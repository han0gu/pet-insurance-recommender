from langchain_core.documents import Document

chunk = Document(
    page_content=('용</td><td>점수</td></tr></thead><tbody><tr><td rowspan="3">검사 소견</td><td>양측 '
 '전정기능 소실</td><td>14</td></tr><tr><td>양측 전정기능 '
 '감소</td><td>10</td></tr><tr><td>일측 전정기능 소실</td><td>4</td></tr><tr><td '
 'rowspan="4">치료 병력</td><td>장기 통원치료(1년간 12회이상)</td><td>6</td></tr><tr><td>장기 '
 '통원치료(1년간'),
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
