from langchain_core.documents import Document

chunk = Document(
    page_content=('- 장 내에서 일시적으로 사육, 훈련 또는 보호하는 영업을 행하는 시설을 말합니다.\n'
 '- ③ 제1항의 반려견 위탁비용은 위탁1일당 이 특별약관의 보험가입금액을 한도로 합니다.\n'
 '- ④ 제1항의 경우 피보험자가 동일한 질병의 치료를 직접 목적으로 2회 이상 입원한 경우\n'
 '- 이를 1회 입원으로 보아 입원일수를 더합니다.\n'
 '- ⑤ 제1항의 경우 피보험자가 병원 또는 의원을 이전하여 입원한 경우에도 동일한 질병의\n'
 '- 치료를 직접 목적으로 입원한 경우에는 계속하여 입원한 것으로 보아 각 입원일수를\n'
 '- 더합니다.'),
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
