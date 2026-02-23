from langchain_core.documents import Document

chunk = Document(
    page_content=('. 회사가 이 계약을 청약할 때 피보험자의 건강상태를 판단할 수 있는 기초자료<br>(건강진단서 사본 등)에 따라 승낙한 경우에 '
 "건강진단서 사본 등에 명기되어<br>있는 사항으로 보험금 지급사유가 발생하였을 때(계약자 또는 피보험자가 회사</p><p id='153' "
 "data-category='paragraph' style='font-size:18px'>- 60 -</p><p id='154' "
 "data-category='list' style='font-size:16px'>에 제출한 기초자료의 내용 중 중요사항을 고의로 사실과 "
 '다르게'),
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
