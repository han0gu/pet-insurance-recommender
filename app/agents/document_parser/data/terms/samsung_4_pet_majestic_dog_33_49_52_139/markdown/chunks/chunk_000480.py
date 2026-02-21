from langchain_core.documents import Document

chunk = Document(
    page_content=('- 아닌 경우에는 본인의 인감증명서 또는 안전성과 신뢰성이 확보된 전자적 수단을\n'
 '- 활용한 보험수익자 의사표시의 확인방법 포함)\n'
 '# 6. 기타 보험수익자가 보험금의 수령에 필요하여 제출하는 서류# ② 제1항 제4호의 사고증명서는 수의사법 제2조(정의)에서 규정한 '
 '동물병원에서 수의사\n'
 '가 발급한 것이어야 합니다.<수의사법 제2조(정의)>- 이 법에서 사용하는 용어의 뜻은 다음과 같다.\n'
 '- 1. "수의사"란 수의업무를 담당하는 사람으로서 농림축산식 품부장관의 면허를 받은 사람을 말\n'
 '- 한다.'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
