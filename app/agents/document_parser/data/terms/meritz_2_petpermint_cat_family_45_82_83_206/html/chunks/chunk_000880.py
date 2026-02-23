from langchain_core.documents import Document

chunk = Document(
    page_content=('피부염</td></tr><tr><td>LAA016</td><td>족피부염</td></tr><tr><td>LAA017</td><td>꼬리샘 '
 '과증식</td></tr><tr><td>LAA018</td><td>발톱 '
 '주위염</td></tr><tr><td>LAA019</td><td>옴진드기 · 개선충</td></tr><tr><td>LAA020 '
 'LAA022</td><td>벼룩 / 진드기 등 외부 기생충 질환 기타 피부 '
 '질환</td></tr><tr><td>QCA001</td><td>귀 가려움증 (원인'),
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
