from langchain_core.documents import Document

chunk = Document(
    page_content=('- 회사 및 보험관련단체 등에 개인정보를 제공할 수 있습니다.\n'
 '- ② 회사는 계약과 관련된 개인정보를 안전하게 관리하여야 합니다.\n'
 '# 제38조(준거법)이 계약은 대한민국 법에 따라 규율되고 해석되며, 약관에서 정하지 않은 사항은 「금융소비자보호에\n'
 '관한 법률」 , 상법, 민법 등 관계 법령을 따릅니다.# 제39조(예금보험에 의한 지급보장)회사가 파산 등으로 인하여 보험금 등을 '
 '지급하지 못할 경우에는 예금자보호법에서 정하는 바에 따라'),
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
