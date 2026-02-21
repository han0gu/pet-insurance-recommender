from langchain_core.documents import Document

chunk = Document(
    page_content=('| 창상봉합술Ⅱ (급여) (안면/경부 |  |  |\n'
 '| 창상봉합술Ⅱ (급여) (안면/경부 | 주: 길이 10cm이상 창상봉합술을 시행할경우 소 정점수에 103.14점을 가산하며, 창상봉합 '
 '길 이가 10cm 증가될때마다 103.14점을 추가 가 산한다. | SC040 |\n'
 '별표11 2대호흡계특정질환 분류표\n'
 '공\n'
 '\uf000 약관에 규정하는 2대호흡계특정질환으로 분류되는 질병은 제9차 개정 한국표준질\n'
 '통\n'
 '병 ․사인분류(KCD, 통계청 고시 제2025-299호, 2026.1.1. 시행) 중 다음에 적은 질'),
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
