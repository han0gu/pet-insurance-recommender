from langchain_core.documents import Document

chunk = Document(
    page_content=('- 카이다이빙, 스쿠버다이빙, 행글라이딩, 수상보트, 패러글라이딩\n'
 '- 2. 모터보트, 자동차 또는 오토바이에 의한 경기, 시범, 흥행(이를 위한 연습을\n'
 '- 포함합니다) 또는 시운전(다만, 공용도로상에서 시운전을 하는 동안 보험금\n'
 '- 지급사유가 발생한 경우에는 보장합니다)'),
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
