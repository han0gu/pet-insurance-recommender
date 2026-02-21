from langchain_core.documents import Document

chunk = Document(
    page_content=("id='150' data-category='paragraph' style='font-size:14px'>관하여 분쟁이 있는 경우 분쟁 "
 "당사자 또는 기타 이해관계인과 회사는 금</p><br><p id='151' data-category='list' "
 "style='font-size:14px'>융감독원장에게 조정을 신청할 수 있으며, 분쟁조정 과정에서 계약자는 관계 법령<br>이 정하는 "
 '바에 따라 회사가 기록 및 유지･관리하는 자료의 열람(사본의 제공 또<br>는 청취를 포함한다)을 요구할 수 있습니다.<br>\uf000 '
 '회사는 일반금융소비자인'),
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
