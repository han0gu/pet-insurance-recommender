from langchain_core.documents import Document

chunk = Document(
    page_content=('수 있으며, 이 경우 회사는 기존계약의 가입 이후 발생한 반려동물의 상해 또는 질병\n'
 '을 사유로 가입을 거절할 수 없습니다. 단, 특별약관 일반사항의 제19조(특별약관의\n'
 '성립) 제1항 및 제2항에도 불구하고 제2항에서 말하는 별도의 반려동물보험 상품으로\n'
 '체결될 수 있습니다.- 1. 재가입일에 있어서 반려동물의 나이가 회사가 최초가입 당시 정한 재가입 나이의\n'
 '- 범위 내일 것\n'
 '- 2. 재가입 전 계약의 보험료가 정상적으로 납입완료 되었을 것\n'
 '- ② 이 계약의 보험기간 종료 후 계약자가 재가입을 원하는 경우 계약자는 재가입 시점에'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
