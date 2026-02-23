from langchain_core.documents import Document

chunk = Document(
    page_content=('| 5 | AFC013 | 상세미상의 피부 신생물 (양성 또는 악성이 불확실한) |  |\n'
 '| 5 | AFA014 | 기타 피부 신생물 (양성) |  |\n'
 '| 5 | AFB014 | 기타 피부 신생물 (악성) |  |\n'
 '| 5 | AFC014 | 기타 피부 신생물 (양성 또는 악성이 불확실 한) |  |\n'
 '| 5 | GAA001 | 외이도염 (세균성) |  |\n'
 '| 5 | GAA002 GAA003 | 외이도염 (말라세지아) 외이도염 (알러지성) |  |\n'
 '171| 구 분 | 특정질병 | 분류코드 | 항목명 |'),
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
