from langchain_core.documents import Document

chunk = Document(
    page_content=('- 「개인정보 보호법」 , 「신용정보의 이용 및 보호에 관한 법률」 등 관계 법령에 정한\n'
 '- 경우를 제외하고 계약자, 피보험자 또는 보험수익자의 동의없이 수집, 이용, 조회 또\n'
 '- 는 제공하지 않습니다. 다만, 회사는 이 계약의 체결, 유지, 보험금 지급 등을 위하여\n'
 '- 위 관계 법령에 따라 계약자 및 피보험자의 동의를 받아 다른 보험회사 및 보험관련\n'
 '- 단체 등에 개인정보를 제공할 수 있습니다.\n'
 '- ② 회사는 계약과 관련된 개인정보를 안전하게 관리하여야 합니다.'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
