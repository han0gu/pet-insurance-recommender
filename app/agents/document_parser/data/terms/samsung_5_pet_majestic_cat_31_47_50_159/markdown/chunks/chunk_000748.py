from langchain_core.documents import Document

chunk = Document(
    page_content=('| 치료 병력 | 단기 통원치료(6개월간 6회미만) | 0 |\n'
 '| 기능 장해 소견 | 두 눈을 감고 일어서기 곤란하거나 두 눈을 뜨고 10m 거리를 직선으 로 걷다가 쓰러지는 경우 | 20 |\n'
 '| 기능 장해 소견 | 두 눈을 뜨고 10m 거리를 직선으로 걷다가 중간에 균형을 잡으려 멈 추어야 하는 경우 | 12 |\n'
 '| 기능 장해 소견 | 두 눈을 뜨고 10m 거리를 직선으로 걸을 때 중앙에서 60cm 이상 벗 어나는 경우 | 8 |\n'
 '2) 평형기능의 장해는 장해판정 직전 1년 이상 지속적인 치료 후 장해가 고착되었'),
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
