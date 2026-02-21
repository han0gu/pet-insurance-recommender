from langchain_core.documents import Document

chunk = Document(
    page_content=('(원인 불명)</td></tr><tr><td>QEA004</td><td>복수 (원인 '
 '불명)</td></tr><tr><td>QEA005</td><td>위장관 출혈(혈토, 혈변)</td></tr><tr><td '
 'rowspan="16">7</td><td rowspan="16">치아 및 구강 질환</td><td>AAA001</td><td>구강 내 '
 '양성 신생물</td></tr><tr><td>AAB001</td><td>구강 내 악성 '
 '신생물</td></tr><tr><td>AAC001</td><td>구강 내 신생물(양성 또는 악성이'),
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
