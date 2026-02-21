from langchain_core.documents import Document

chunk = Document(
    page_content=('- 가) 한 눈의 안구(눈동자)의 주시야(머리를 움직이지 않고 눈만을 움직여서 볼\n'
 '- 수 있는 범위)의 운동범위가 정상의 1/2 이하로 감소된 경우\n'
 '- 나) 중심 20도 이내에서 복시(물체가 둘로 보이거나 겹쳐 보임)를 남긴 경우\n'
 '- 7) "안구(눈동자)의 뚜렷한 조절기능장해" 라 함은 조절력이 정상의 1/2 이하로\n'
 '- 감소된 경우를 말한다. 다만, 조절력의 감소를 무시할 수 있는 50세 이상(장해\n'
 '- 진단시 연령 기준)의 경우에는 제외한다.\n'
 '- 8) "뚜렷한 시야 장해" 라 함은 한 눈의 시야 범위가 정상시야 범위의 60% 이하'),
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
