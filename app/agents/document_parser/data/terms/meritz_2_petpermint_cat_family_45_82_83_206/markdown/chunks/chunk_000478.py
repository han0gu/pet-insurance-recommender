from langchain_core.documents import Document

chunk = Document(
    page_content=('| FEA002 | 질환 | 백내장 (우안) |  |\n'
 '| FEA003 FEA004 | 질환 | 수정체 (아) 탈구 백내장 |  |\n'
 '| FFA001 | 질환 | 망막 변성 / 망막 위축 / PRA |  |\n'
 '| FFA002 | 질환 | 망막 박리 (유리체 변성 포함) |  |\n'
 '| FGA001 | 질환 | 녹내장 (좌안) |  |\n'
 '| FGA002 | 질환 | 녹내장 (우안) |  |\n'
 '| FGA003 | 질환 | 동양안충증 |  |\n'
 '| FGA004 | 질환 | 기타 안과 질환 |  |\n'
 '| FGA005 | 질환 | 초자체변성 |  |'),
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
