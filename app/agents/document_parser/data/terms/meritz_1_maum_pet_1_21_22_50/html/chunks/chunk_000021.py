from langchain_core.documents import Document

chunk = Document(
    page_content=("/ 연간 ( )원</td></tr></tbody></table><br><p id='28' data-category='paragraph' "
 "style='font-size:18px'>【보험금 지급금액 예시】아래의 경우는 이해를 돕기 위한 예시이며, 자기부담금,<br>지급한도 "
 "등은 달라질 수 있습니다.</p><br><p id='29' data-category='list' "
 "style='font-size:18px'>- 보험계약일(보장개시일) : 2025년 5월 1일<br>- 자기부담금: 3만원<br>- "
 '보상비율: 70%<br>- 1일'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
