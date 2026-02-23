from langchain_core.documents import Document

chunk = Document(
    page_content=(". 상당한 이유없이 손해조사를 거부 또는 회피할 때</p><br><p id='99' data-category='paragraph' "
 "style='font-size:14px'>② 제1항 제1호에도 불구하고 다음 중 한가지의 경우에 해당되는 때에는 회사는 "
 "계약을<br>해지할 수 없습니다.</p><br><p id='100' data-category='list' "
 "style='font-size:14px'>1. 회사가 최초계약 체결당시에 그 사실을 알았거나 과실로 알지 못하였을 때<br>2"),
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
