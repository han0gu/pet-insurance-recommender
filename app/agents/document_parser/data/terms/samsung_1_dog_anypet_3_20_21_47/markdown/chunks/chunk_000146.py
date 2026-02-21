from langchain_core.documents import Document

chunk = Document(
    page_content=('- 계약의 보험기간 만료일)까지 보상하여 드립니다.\n'
 '# 제5조(개별계약으로의 전환)- ① 피보험자가 퇴직 등의 사유로 인하여 피보험단체에서 탈퇴하는 경우 피보험자가 보험료의 일부를\n'
 '- 부담한 경우에 한하여 탈퇴일로부터 1개월 이내에 계약자 또는 피보험자는 회사의 승낙을 얻어 개\n'
 '- 별계약으로 전환할 수 있으며, 이 경우 피보험자는 개별계약의 계약자가 됩니다.\n'
 '- ② 제1항에 따라 개별계약으로 전환시에는 전환후 피보험자의 보험기간은 이 계약의 남은 기간으로'),
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
