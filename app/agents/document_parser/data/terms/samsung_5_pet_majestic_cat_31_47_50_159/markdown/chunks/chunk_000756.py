from langchain_core.documents import Document

chunk = Document(
    page_content=('# 4) "씹어먹는 기능에 약간의 장해를 남긴 때" 라 함은 아래의 경우 중 하나 이상\n'
 '에 해당되는 때를 말한다.- 가) 약간의 개구운동 제한 또는 약간의 저작운동 제한으로 부드러운 고형식(밥,\n'
 '- 빵 등)만 섭취 가능한 경우\n'
 '- 나) 위·아래턱(상·하악)의 가운데 앞니(중절치)간 최대 개구운동이 2cm 이하로\n'
 '- 제한되는 경우\n'
 '- 다) 위·아래턱(상·하악)의 부정교합(전방, 측방)이 1cm 이상인 경우\n'
 '- 라) 양측 각 1개 또는 편측 2개 이하의 치아만 교합되는 상태'),
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
