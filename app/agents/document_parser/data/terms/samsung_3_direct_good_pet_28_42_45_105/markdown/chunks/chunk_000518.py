from langchain_core.documents import Document

chunk = Document(
    page_content=('- 된 반려견을 수탁기관에 위탁함으로써 발생한 위탁비용을 반려견 위탁비용으로 보험\n'
 '- 수익자에게 지급합니다. 다만, 반려견 위탁비용의 지급일수는 1회 입원당 180일을 한\n'
 '- 도로 하며, 피보험자의 입원기간을 초과할 수 없습니다.\n'
 '- ② 제1항의 「수탁기관」 이라 함은 동물보호법 시행규칙 제43조(등록영업의 세부 범위)\n'
 '- 에서 정하는 동물위탁관리업자로써, 반려동물 소유자의 위탁을 받아 반려동물을 영업\n'
 '- 장 내에서 일시적으로 사육, 훈련 또는 보호하는 영업을 행하는 시설을 말합니다.'),
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
