from langchain_core.documents import Document

chunk = Document(
    page_content=('- 니다.\n'
 '- ④ 제1항 및 제3항에 따라 계약이 해지된 경우 회사는 제30조(보험료의 환급) 제1항 제1호에 따른 환\n'
 '- 급금을 계약자에게 지급합니다.\n'
 '- ⑤ 계약자는 제1항에 따른 제척기간에도 불구하고 민법 등 관계 법령에서 정하는 바에 따라 법률상의\n'
 '- 권리를 행사할 수 있습니다.\n'
 '【제척기간】 어떤 종류의 권리에 대하여 법률상으로 정하여진 존속기간을 말하며, 이 기간이 지나면 해당'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
