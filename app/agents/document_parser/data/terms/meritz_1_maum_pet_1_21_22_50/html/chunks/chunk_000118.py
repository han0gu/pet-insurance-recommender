from langchain_core.documents import Document

chunk = Document(
    page_content=('. 만약, 회사가 전자우편 및 전자적 의사표시로<br>제공한 경우 계약자 또는 그 대리인이 약관 및 계약자 보관용 청약서 등을 '
 "수신하였을<br>때에는 해당 문서를 드린 것으로 봅니다.</p><br><p id='18' data-category='list' "
 "style='font-size:14px'>1. 서면교부<br>2. 우편 또는 전자우편<br>3"),
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
