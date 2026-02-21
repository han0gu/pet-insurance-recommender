from langchain_core.documents import Document

chunk = Document(
    page_content=('. 계약 체결 전에 피보험자의</td></tr><tr><td>고지사항이 청약서에 제대로 확인하시기 바랍니다.</td><td>기재되어 '
 '있는지 반드시</td></tr><tr><td colspan="2">용 어 풀 이 해지 현재 유지되고 있는 계약 또는 효력이 상실된 계약을 '
 "장래를 향하여 소멸시키 보</td></tr></tbody></table><br><p id='156' "
 "data-category='paragraph' style='font-size:16px'>거나 계약유지 의사를 포기하여 만기일 이전에 "
 '계약관계를 청산하는 것</p><p'),
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
