from langchain_core.documents import Document

chunk = Document(
    page_content=('| 장해 두 눈을 뜨고 10m 거리를 직선으로 걷다가 소견 중간에 균형을 잡으려 멈추어야 하는 경우 | 12 |  |\n'
 '| 두 눈을 뜨고 10m 거리를 직선으로 걸을 때 중앙에서 60cm 이상 벗어나는 경우 | 8 |  |\n'
 '| 2) | 평형기능의 장해는 장해판정 직전 1년 이상 지속적인 치료 | 후 장해가 고 |\n'
 '| 때 | 착되었을 판정하며, 뇌병변 여부, 전정기능 이상 및 장해상태를 평가 하기 위해 아래의 검사들을 기초로 한다. 가) '
 '뇌영상검사(CT, MRI) |\n'
 '| --- | --- |'),
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
