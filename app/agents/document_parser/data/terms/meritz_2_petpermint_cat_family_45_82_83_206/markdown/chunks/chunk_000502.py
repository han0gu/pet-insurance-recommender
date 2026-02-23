from langchain_core.documents import Document

chunk = Document(
    page_content=('| 7 | 치아 및 구강 질환 | JAA001 | 치수염 |\n'
 '| 7 | 치아 및 구강 질환 | JAA002 | 치아 골절 |\n'
 '| 7 | 치아 및 구강 질환 | JAA003 | 애나멜 저형성증 |\n'
 '| 7 | 치아 및 구강 질환 | JAA004 | 유치 잔존증 |\n'
 '| 7 | 치아 및 구강 질환 | JAA005 | 부정 교합 |\n'
 '| 7 | 치아 및 구강 질환 | JAA006 | 기타 치과 질환 |\n'
 '| 7 | 치아 및 구강 질환 | JBA001 | 구내염 / 설염 |\n'
 '| 7 | 치아 및 구강 질환 | JBA002 | 구개열 |'),
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
