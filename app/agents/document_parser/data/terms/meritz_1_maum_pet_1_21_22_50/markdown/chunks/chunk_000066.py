from langchain_core.documents import Document

chunk = Document(
    page_content=('- 활이 이루어진 경우에는 부활계약을 제2항의 최초계약으로 봅니다.(부활(효력회복)이\n'
 '- 여러차례 발생된 경우에는 각각의 부활(효력회복)계약을 최초계약으로 봅니다)\n'
 '- 11 -# 제18조(사기에 의한 계약)계약자 또는 피보험자가 사기에 의하여 계약이 성립되었음을 회사가 증명하는 경우에는 계\n'
 '약일부터 5년 이내(사기사실을 안 날부터 1개월 이내)에 계약을 취소할 수 있습니다.제 4 관 보험계약의 성립과 유지제19조(보험계약의 '
 '성립)- ① 계약은 계약자의 청약과 회사의 승낙으로 이루어집니다.'),
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
