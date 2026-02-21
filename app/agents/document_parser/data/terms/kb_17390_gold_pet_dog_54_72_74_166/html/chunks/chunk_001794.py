from langchain_core.documents import Document

chunk = Document(
    page_content=('진폐증</td><td>J64</td></tr><tr><td>결핵과 연관된 진폐증</td><td>J65</td></tr><tr><td>특정 '
 '유기물먼지에 의한 기도질환</td><td>J66 보</td></tr><tr><td>유기물먼지에 의한 과민성 폐렴 화학물질, 가스, 훈증기 '
 '및</td><td>J67 통약</td></tr><tr><td>물김의 흡입에 의한 호흡기 병태</td><td>J68 '
 '관</td></tr><tr><td>고체 및 액체에 의한 폐렴</td><td>J69</td></tr><tr><td '
 'rowspan="3">중금속에'),
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
