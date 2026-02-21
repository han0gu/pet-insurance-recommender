from langchain_core.documents import Document

chunk = Document(
    page_content=('치과 질환</td></tr><tr><td>JBA001</td><td>구내염 / '
 '설염</td></tr><tr><td>JBA002</td><td>구개열</td></tr><tr><td>JBA003</td><td>침샘 질환 '
 '(침샘염 / 점액 낭종 / 하마종)</td></tr><tr><td>JBA004</td><td>치은염 / '
 '치주염</td></tr><tr><td>JBA005 JBA006</td><td>치근농양 / '
 '근첨농양</td></tr><tr><td>JBA007</td><td>기타 구강 질환'),
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
