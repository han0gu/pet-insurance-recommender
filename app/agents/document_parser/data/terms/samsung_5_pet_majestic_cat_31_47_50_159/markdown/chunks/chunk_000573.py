from langchain_core.documents import Document

chunk = Document(
    page_content=('를 따릅니다. 이 경우 부활(효력회복)일을 보험계약일로 하여 제1조(보험금의 지급사유)\n'
 '제3항을 적용합니다.# 제8조 (준용규정)이 추가특별약관에 정하지 않은 사항은 4-1. 반려묘 의료비(치과및구강질환포함)(재가입\n'
 '형) 특별약관을 따르며, 4-1. 반려묘 의료비(치과및구강질 환포 함)(재가입형) 특별약관에서\n'
 '정하지 않은 사항은 특별약관 일반사항을 따릅니다. 특별약관 일반사항에서도 정하지 않'),
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
