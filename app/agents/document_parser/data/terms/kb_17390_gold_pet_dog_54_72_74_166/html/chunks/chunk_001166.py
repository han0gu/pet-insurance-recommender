from langchain_core.documents import Document

chunk = Document(
    page_content=('. 그러나 피보험자의 책임있는<br>사유로 지체된 경우에는 그 해당기간에 대한 이자를 더하여 지급하지 않습니다.<br>다만, 회사는 '
 "피보험자가 분쟁조정을 신청했다는 사유만으로 이자지급을 거절하<br>지 않습니다.</p><br><p id='176' "
 "data-category='paragraph' style='font-size:14px'>제9조(보험금 등의 지급한도)</p><br><h1 "
 "id='177' style='font-size:14px'>회사는 1회의 보험사고에</h1><br><p id='178'"),
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
