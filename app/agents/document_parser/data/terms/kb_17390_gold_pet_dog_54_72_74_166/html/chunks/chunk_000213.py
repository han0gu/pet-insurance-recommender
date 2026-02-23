from langchain_core.documents import Document

chunk = Document(
    page_content=('colspan="3">어떤 사람의 생사불명의 상태가 일정기간 이상 계속 될 때 이해관계가 있는 사람 약 관 의 청구에 의해 사망한 것으로 '
 '인정하고 신분이나 재산에 관한 모든 법적 관계를 확정시키는 법원의 결정을 '
 "말합니다.</td></tr></tbody></table><br><table id='23' "
 "style='font-size:14px'><thead></thead><tbody><tr><td>부 가 설 명</td><td>민법 "
 '제27조(실종의 선고)</td></tr><tr><td colspan="2">1'),
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
