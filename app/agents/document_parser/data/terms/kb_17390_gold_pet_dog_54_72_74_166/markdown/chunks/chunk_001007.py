from langchain_core.documents import Document

chunk = Document(
    page_content=('| 특정세균성 | 폐렴연쇄알균에 의한 폐렴 폐렴 | J13 |\n'
 '| 특정외부요인 폐질환 | 인플루엔자균에 의한 폐렴 화학물질, 가스, 훈증기 및 물김의 | J14 |\n'
 '| 특정외부요인 폐질환 | 흡입에 의한 호흡기병태 | J68 |\n'
 '| 하부호흡기 | 고체 및 액체에 의한 폐렴 | J69 |\n'
 '| 하부호흡기 | 폐기종 | J43 |\n'
 '|  | 특정질환 기관지확장증 | J47 |\n'
 '| 달리 분류되지 않은 흉막삼출액 | J90 |  |\n'
 '| 흉막판 | J92 |  |\n'
 '| 흉막특정질환 | 기타 흉막의 병태 | J94 |'),
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
