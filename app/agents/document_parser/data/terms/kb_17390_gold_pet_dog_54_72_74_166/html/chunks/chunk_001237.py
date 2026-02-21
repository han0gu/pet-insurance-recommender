from langchain_core.documents import Document

chunk = Document(
    page_content=("안에서 회사에</td><td>보험금의 지급을 청구할 수 있습니다.</td></tr></tbody></table><br><p id='44' "
 "data-category='paragraph' style='font-size:16px'>용 어 풀 이 타인을 위한 계약<br>계약자가 "
 "타인의 이익을 위하여 자기의 이름으로 체결하는 보험계약을 말합</p><br><p id='45' "
 "data-category='paragraph' style='font-size:16px'>니다.</p><p id='46' "
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
