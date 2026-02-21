from langchain_core.documents import Document

chunk = Document(
    page_content=('| 5 | 피부질환 | AFA014 | 기타 피부 신생물 (양성) |\n'
 '| 5 | 피부질환 | AFB014 | 기타 피부 신생물 (악성) |\n'
 '| 5 | 피부질환 | AFC014 | 기타 피부 신생물 (양성 또는 악성이 불 확실한) |\n'
 '| 5 | 피부질환 | GAA001 | 외이도염 (세균성) |\n'
 '197| 구 분 | 특정질병 | 분류코드 | 항목명 |\n'
 '| --- | --- | --- | --- |\n'
 '|  |  | GAA002 | 외이도염 (말라세지아) |\n'
 '| GAA003 | 외이도염 (알러지성) |  |  |'),
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
