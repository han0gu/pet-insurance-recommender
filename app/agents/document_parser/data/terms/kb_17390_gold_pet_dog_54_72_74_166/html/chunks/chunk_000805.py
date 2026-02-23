from langchain_core.documents import Document

chunk = Document(
    page_content=(". 보험금 지급사유에 대해 제3자의 의견에 따르기로 한 경우</p><br><table id='172' "
 "style='font-size:16px'><thead></thead><tbody><tr><td>용 어 풀</td><td>이 분쟁조정 "
 '신청</td></tr><tr><td colspan="2">분쟁조정 신청은 이 약관의 ｢분쟁의 조정｣ 조항에 따르며 분쟁조정 신청 대상기 '
 "관은 금융감독원의 금융분쟁조정위원회를 말합니다.</td></tr></tbody></table><br><p id='173' "
 "data-category='paragraph'"),
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
