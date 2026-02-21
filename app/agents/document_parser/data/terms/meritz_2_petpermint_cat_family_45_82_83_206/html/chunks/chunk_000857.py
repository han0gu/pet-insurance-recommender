from langchain_core.documents import Document

chunk = Document(
    page_content=('(홍채염 / 전안방 출혈 포함)</td></tr><tr><td>FEA001</td><td>백내장 '
 '(좌안)</td></tr><tr><td>FEA002</td><td>백내장 (우안)</td></tr><tr><td>FEA003 '
 'FEA004</td><td>수정체 (아) 탈구 백내장</td></tr><tr><td>FFA001</td><td>망막 변성 / 망막 위축 '
 '/ PRA</td></tr><tr><td>FFA002</td><td>망막 박리 (유리체 변성 '
 '포함)</td></tr><tr><td>FGA001</td><td>녹내장'),
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
