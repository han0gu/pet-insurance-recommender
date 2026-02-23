from langchain_core.documents import Document

chunk = Document(
    page_content=('- 등에 개인정보를 제공할 수 있습니다.\n'
 '- ② 회사는 계약과 관련된 개인정보를 안전하게 관리하여야 합니다.\n'
 '# 제41조(준거법)이 계약은 대한민국 법에 따라 규율되고 해석되며, 약관에서 정하지 않은 사항은 「금융소\n'
 '비자보호에 관한 법률」, 상법, 민법 등 관계 법령을 따릅니다.# 제42조(예금보험에 의한 지급보장)회사가 파산 등으로 인하여 보험금 '
 '등을 지급하지 못할 경우에는 예금자보호법에서 정하는\n'
 '바에 따라 그 지급을 보장합니다.# 【예금자보호제도】예금자보호제도란 예금보험공사가 평소에 금융기관으로 부터 보험료를 받아 기금을 적'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
