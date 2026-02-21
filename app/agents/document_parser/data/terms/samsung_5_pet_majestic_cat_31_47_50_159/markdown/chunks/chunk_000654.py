from langchain_core.documents import Document

chunk = Document(
    page_content=('하는 바에 따라 회사가 적립한 사망당시 이 추가특별약관의 계약자적립액 및 미경과보험\n'
 '료를 계약자에게 지급하고, 이 추가특별약관은 더 이상 효력이 없습니다.# 제9조 (준용규정)이 추가특별약관에 정하지 않은 사항은 4-1. '
 '반려묘 의료비(치과및구강질환포함)(재가입\n'
 '형) 특별약관을 따르며, 4-1. 반려묘 의료비(치과및구강질환포함)(재가입형) 특별약관에서\n'
 '정하지 않은 사항은 특별약관 일반사항을 따릅니다. 특별약관 일반사항에서도 정하지 않\n'
 '은 사항은 보통약관을 따릅니다. 다만, 보통약관 제10조(환급금의 중도인출), 제11조(만'),
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
