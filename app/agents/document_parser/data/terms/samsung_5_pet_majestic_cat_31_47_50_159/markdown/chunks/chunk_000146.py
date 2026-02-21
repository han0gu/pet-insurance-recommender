from langchain_core.documents import Document

chunk = Document(
    page_content=('사례에 비추어 보험수익자에게 매우 불합리하게 합의를 하는 것을 의미합니다.# 제45조 (개인정보보호)- ① 회사는 이 계약과 관련된 '
 '개인정보를 이 계약의 체결, 유지, 보험금 지급 등을 위하여\n'
 '- 「개인정보 보호법」,「신용정보의 이용 및 보호에 관한 법률」등 관계 법령에 정한\n'
 '- 경우를 제외하고 계약자, 피보험자 또는 보험수익자의 동의없이 수집, 이용, 조회 또\n'
 '- 는 제공하지 않습니다. 다만, 회사는 이 계약의 체결, 유지, 보험금 지급 등을 위하여'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
