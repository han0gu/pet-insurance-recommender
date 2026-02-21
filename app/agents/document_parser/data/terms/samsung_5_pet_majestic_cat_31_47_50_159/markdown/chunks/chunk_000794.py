from langchain_core.documents import Document

chunk = Document(
    page_content=('- 나) 근전도 검사상 불완전한 손상(incomplete injury)소견이 있으면서 도수근력\n'
 '- 검사(MMT)에서 근력이 "3등급(fair)" 인 경우\n'
 '11) "가관절주)이 남아 뚜렷한 장해를 남긴 때" 라 함은 상완골에 가관절이 남은\n'
 '경우 또는 요골과 척골의 2개 뼈 모두에 가관절이 남은 경우를 말한다.\n'
 '주) 가관절이란, 충분한 경과 및 골이식술 등 골유합을 얻는데 필요한 수술적\n'
 "치료를 시행하였음에도 불구하고 골절부의 유합이 이루어지지 않는 '불유"),
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
