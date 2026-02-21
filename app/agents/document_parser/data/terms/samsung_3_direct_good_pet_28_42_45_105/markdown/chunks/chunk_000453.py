from langchain_core.documents import Document

chunk = Document(
    page_content=('- 아닌 경우에는 본인의 인감증명서, 본인서명사실확인서 또는 안전성과 신뢰성이\n'
 '- 확보된 전자적 수단을 활용한 보험수익자 의사표시의 확인방법 포함)\n'
 '- 6. 기타 보험수익자가 보험금의 수령에 필요하여 제출하는 서류\n'
 '② 제1항 제4호의 사고증명서는 수의사법 제2조(정의)에서 규정한 동물병원에서 수의사\n'
 '가 발급한 것이어야 합니다.<관련법규># [수의사법 제2조(정의)]- 1. "수의사"란 수의업무를 담당하는 사람으로서 농림축산식품부장관의 '
 '면허를 받은 사람을 말한다.'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
