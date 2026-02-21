from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:14px'>질</p><p id='50' data-category='paragraph' "
 "style='font-size:16px'>\uf000 보험증권에 기재된 반려동물이 보험기간 중에 사망하여 보험의 목적에 대해 "
 "이</p><br><p id='51' data-category='list' style='font-size:16px'>특별약관에서 정한 "
 '보험금 지급사유가 더 이상 발생할 수 없는 경우에는 이 특별<br>반<br>약관 계약도 소멸되며 회사는 "보험료 및 해약환급금 '
 '산출방법서"에서 정하는 바<br>에 따라'),
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
