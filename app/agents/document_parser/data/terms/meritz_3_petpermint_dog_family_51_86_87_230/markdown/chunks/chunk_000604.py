from langchain_core.documents import Document

chunk = Document(
    page_content=('| 치료 병력 | 단기 통원치료(6개월간 6회미만) | 0 |\n'
 '| 기능 장해 소견 | 두 눈을 감고 일어서기 곤란하거나 두 눈 을 뜨고 10m 거리를 직선으로 걷다가 쓰 | 20 |\n'
 '| 기능 장해 소견 | 경우 | 12 |\n'
 '| 기능 장해 소견 | 러지는 두 눈을 뜨고 10미터 거리를 직선으로 걷 다가 중간에 균형을 잡으려 멈추어야 하 는 경우 두 눈을 뜨고 '
 '10m 거리를 직선으로 걸을 때 중앙에서 60cm 이상 벗어나는 경우 | 8 |\n'
 '2) 평형기능의 장해는 장해판정 직전 1년 이상 지속적인'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
