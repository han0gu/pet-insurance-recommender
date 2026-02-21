from langchain_core.documents import Document

chunk = Document(
    page_content=('형이나 기능장해가 발생하여 그 원상회복을 목적으로 사고일로부터 2년 이내에\n'
 '성형외과 전문의로부터 성형수술을 받은 경우 아래와 같이 최대 수술길이에 따라\n'
 '상해흉터복원수술비Ⅱ(안면부)를 보험수익자에게 매 사고시마다 지급합니다.구 분지급금액| 안면부(5cm이상~ 10cm미만) | 가입금액의 '
 '60% |\n'
 '| --- | --- |\n'
 '| 안면부(10cm이상) | 가입금액의 100% |\n'
 '| 제1항에서 정한 안면부란 이마를 포함하여 | 목까지의 얼굴부분을 말합니다. |'),
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
