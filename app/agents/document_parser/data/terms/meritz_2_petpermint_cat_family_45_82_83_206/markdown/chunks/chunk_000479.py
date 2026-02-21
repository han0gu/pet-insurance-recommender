from langchain_core.documents import Document

chunk = Document(
    page_content=('| FGA005 | 질환 | 초자체변성 |  |\n'
 '| FGA006 | 질환 | 상공막염 |  |\n'
 '| FGA007 | 질환 | 녹내장 |  |\n'
 '| FGA008 | 질환 | 고양이 호산구성 각결막염 |  |\n'
 '| QBA001 | 질환 | 눈곱 (원인 불명) |  |\n'
 '| QBA002 | 질환 | 결막 충혈 (원인 불명) |  |\n'
 '| QBA003 | 질환 | 눈 가려움증 (원인 불명) |  |\n'
 '| 3 | 순환기 질환 | ACA001 | 순환기 계통의 양성 신생물 |\n'
 '| 3 | 순환기 질환 | ACB001 | 순환기 계통의 악성 신생물 |'),
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
