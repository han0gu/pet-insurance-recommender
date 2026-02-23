from langchain_core.documents import Document

chunk = Document(
    page_content=('| 간질영향 호흡기질환 | 달리 분류되지 않은 폐호산구증가 | 통약 J82 |\n'
 '| 하기도 화농· | 기타 간질성 폐질환 J84 | 관 |\n'
 '| 하기도 화농· | 폐 및 종격의 농양 | J85 |\n'
 '# 괴사성 질환농흉J86주) 1. 대상질병 분류표의 분류번호와 다르나 한국표준질병․사인분류의 기준에 따\n'
 '라 분류번호를 동시에 부여가 가능한 경우 대상질병 분류에 포함합니다. 특별\n'
 '2. 제10차 개정 이후 이 약관에서 보장하는 2대호흡계특정질환 해당여부는 피 약\n'
 '보험자가 진단된 당시 시행되고 있는 한국표준질병․사인분류에 따라 판단 관\n'
 '합니다.'),
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
