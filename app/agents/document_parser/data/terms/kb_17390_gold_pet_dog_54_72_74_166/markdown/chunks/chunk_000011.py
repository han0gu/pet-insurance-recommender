from langchain_core.documents import Document

chunk = Document(
    page_content=('| 보험기간 | 계약에 따라 보장을 받는 기간을 말합니다. 정상적으로 영업하는 날을 |\n'
 '| 영업일 | 회사가 영업점에서 말하며, 토요 일, "관공서의 공휴일에 관한 규정"에 따른 공휴일과 노동 절을 제외합니다. |\n'
 '|  | (대통령령 제31930호) |\n'
 '| --- | --- |\n'
 '# 관 련 법 규# 관공서의 공휴일에 관한 규정 제2조 및 제3조통\n'
 '제2조(공휴일)\n'
 '관공서의 공휴일은 다음 각 호와 같다. 다만, 재외공관의 공휴일은 우리나 사항# 라의- 국경일 중 공휴일과 주재국의 공휴일로 한다.\n'
 '- 1. 일요일'),
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
