from langchain_core.documents import Document

chunk = Document(
    page_content=('170| 구 분 | 특정질병 | 분류코드 | 항목명 |\n'
 '| --- | --- | --- | --- |\n'
 '|  | 피부질환 | AGA004 | 기타 비뇨기계 양성 신생물 |\n'
 '| AGB004 | 피부질환 | 기타 비뇨기계 악성 신생물 |  |\n'
 '| AGC004 | 피부질환 | 기타 비뇨기계 신생물 (양성 또는 악성이 불 확실한) |  |\n'
 '| OAA001 | 피부질환 | 급성 신부전 |  |\n'
 '| OAA002 | 피부질환 | 신우 신염 |  |\n'
 '| OAA003 | 피부질환 | 수신증 |  |'),
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
