from langchain_core.documents import Document

chunk = Document(
    page_content=('람을 말합니다.# [심신박약자(心神薄弱者)]심신상실의 상태까지는 이르지 않았으나, 마음이나 정신의 장애로 인하여 사물을 변별할 능력이나\n'
 '의사를 결정할 능력이 미약한 사람을 말합니다.3. 계약을 체결할 때 계약에서 정한 피보험자의 나이에 미달되었거나 초과되었을 경\n'
 '우. 다만, 회사가 나이의 착오를 발견하였을 때 이미 계약나이에 도달한 경우에는\n'
 '유효한 계약으로 보나, 제2호의 만 15세 미만자에 관한 예외가 인정되는 것은 아'),
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
