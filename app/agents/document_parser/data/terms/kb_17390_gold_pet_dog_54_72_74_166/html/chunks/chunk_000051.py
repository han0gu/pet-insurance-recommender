from langchain_core.documents import Document

chunk = Document(
    page_content=('. 보험금</td><td>지급사유에 대해 제3자의 의견에 따르기로 한 경우</td></tr><tr><td colspan="2">용 어 풀 '
 '이 분쟁조정 신청 분쟁조정 신청은 이 약관의 ｢분쟁의 조정｣ 조항에 따르며 분쟁조정 신청 '
 "대상기</td></tr></tbody></table><br><p id='66' data-category='list' "
 "style='font-size:16px'>관은 금융감독원의 금융분쟁조정위원회를 말합니다.<br>\uf000 제2항에 의하여 장해지급률의 "
 '판정 및 지급할 보험금의 결정과 관련하여 확정된 장<br>해지급률에'),
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
