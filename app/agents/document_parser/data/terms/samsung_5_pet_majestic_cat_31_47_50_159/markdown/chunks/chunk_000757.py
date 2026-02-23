from langchain_core.documents import Document

chunk = Document(
    page_content=('- 라) 양측 각 1개 또는 편측 2개 이하의 치아만 교합되는 상태\n'
 '- 마) 연하기능검사(비디오 투시검사)상 연하장애가 있고, 유동식 섭취시 간헐적\n'
 '- 으로 흡인이 발생하고 부드러운 고형식 외에는 섭취가 불가능한 상태\n'
 '- 5) 개구장해는 턱관절의 이상으로 개구운동 제한이 있는 상태를 말하며, 최대 개구\n'
 '- 상태에서 위·아래턱(상·하악)의 가운데 앞니(중절치)간 거리를 기준으로 한다.\n'
 '- 단, 가운데 앞니(중절치)가 없는 경우에는 측정가능한 인접 치아간 거리의 최\n'
 '- 대치를 기준으로 한다.'),
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
