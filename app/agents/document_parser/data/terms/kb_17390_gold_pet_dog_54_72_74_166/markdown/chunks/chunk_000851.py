from langchain_core.documents import Document

chunk = Document(
    page_content=('| 양측 전정기능 소실 | 14 |  |\n'
 '| 양측 전정기능 감소 소견 | 10 |  |\n'
 '|  | 일측 전정기능 소실 장기 통원치료(1년간 12회이상) | 4 6 |\n'
 '| 치료 장기 통원치료(1년간 6회이상) | 4 |  |\n'
 '| 병력 단기 통원치료(6개월간 6회이상) | 2 |  |\n'
 '| 단기 통원치료(6개월간 6회미만) | 0 |  |\n'
 '|  | 두 눈을 감고 일어서기 곤란하거나 두 눈을 뜨고 10m 거리를 직선으로 걷다가 쓰러지는 경우 기능 | 20 |'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
