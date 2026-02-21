from langchain_core.documents import Document

chunk = Document(
    page_content=('- 3. 신분증(주민등록증이나 운전면허증 등 사진이 붙은 정부기관발행 신분증, 본인\n'
 '- 이 아닌 경우에는 본인의 인감증명서, 본인서명사실확인서 또는 안전성과 신\n'
 '- 뢰성이 확보된 전자적 수단을 활용한 보험수익자 의사표시의 확인방법 포함)\n'
 '- 4. 기타 보험수익자가 보험금의 수령에 필요하여 제출하는 서류\n'
 '용 어 풀 이 건강보험심사평가원 진료수가코드(EDI)「건강보험 행위 급여․비급여 목록 및 급여 상대가치점수(보건복지부 고시)」에서 정한 '
 '처치 및 수술료, 검사료, 방사선 치료료 등을 포함한 항목에 부여되'),
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
