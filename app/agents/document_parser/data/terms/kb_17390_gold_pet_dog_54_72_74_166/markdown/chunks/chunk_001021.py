from langchain_core.documents import Document

chunk = Document(
    page_content=('| 외부요인에 의한 폐질환 | 특정 유기물먼지에 의한 기도질환 | J66 보 |\n'
 '| 외부요인에 의한 폐질환 | 유기물먼지에 의한 과민성 폐렴 화학물질, 가스, 훈증기 및 | J67 통약 |\n'
 '| 외부요인에 의한 폐질환 | 물김의 흡입에 의한 호흡기 병태 | J68 관 |\n'
 '| 외부요인에 의한 폐질환 | 고체 및 액체에 의한 폐렴 | J69 |\n'
 '| 중금속에 의한 질환 | 기타 외부요인에 의한 호흡기 병태 약물 및 중금속 유발 세뇨관-간질 및 | J70 N14 특별 |'),
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
