from langchain_core.documents import Document

chunk = Document(
    page_content=('| FBA002 | 각막 이영양증 |  |  |\n'
 '| FBA003 | 기타 각막염 (판누스 포함) |  |  |\n'
 '| FBA004 | 각막염(비궤양성) |  |  |\n'
 '| FCA001 | 건성 각결막염 · KCS |  |  |\n'
 '| FCA002 | 결막염 (결막 부종 포함) |  |  |\n'
 '| FDA001 | 포도막염 (홍채염 / 전안방 출혈 포함) |  |  |\n'
 '| FEA001 | 백내장 (좌안) |  |  |\n'
 '| FEA002 FEA003 | 백내장 (우안) 수정체 (아) 탈구 |  |  |\n'
 '| FEA004 | 백내장 |  |  |'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
