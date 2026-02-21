from langchain_core.documents import Document

chunk = Document(
    page_content=('의원 및 조산원으로 나누어집니다.# 제10조 (보험금 등의 지급절차)- ① 회사는 제9조(보험금 등의 청구)에서 정한 서류를 접수한 '
 '때에는 접수증을 드리고 휴\n'
 '- 대전화 문자메시지 또는 전자우편 등으로 송부하며, 그 서류를 접수한 날부터 3영업\n'
 '- 일 이내에 보험금을 지급하거나 보험료의 납입을 면제합니다.\n'
 '- ② 회사가 보험금 지급사유 또는 보험료 납입면제 사유를 조사ㆍ확인하기 위해 필요한\n'
 '- 기간이 제1항의 지급기일을 초과할 것이 명백히 예상되는 경우에는 그 구체적 사유와'),
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
