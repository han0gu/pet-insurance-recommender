from langchain_core.documents import Document

chunk = Document(
    page_content=('보상한도액 (30만원)} 중 높은 금액 = 100만원 상 예시② 해 ·MRI/CT에 대한 연간 지급한도(연간 1회한)가 모두 소진된 '
 "경우</td></tr></tbody></table><br><p id='229' data-category='paragraph' "
 "style='font-size:14px'>·최대 보상한도액 = 30만원(항암약물치료 보상한도액 적용)<br>\uf000 보험수익자와 "
 '회사가 제1조(보험금의 지급사유)의 보험금 지급사유에 대해 합의<br>하지 못할 때는 보험수익자와 회사가 함께 제3자를 정하고 그 '
 '제3자의 의견에'),
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
