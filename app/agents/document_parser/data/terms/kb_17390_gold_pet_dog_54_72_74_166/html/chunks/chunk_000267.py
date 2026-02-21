from langchain_core.documents import Document

chunk = Document(
    page_content=("수락여부를 계약자에 통지하여야 하<br>며, 거절할 때에는 거절 사유를 함께 통지하여야 합니다.</p><br><p id='75' "
 "data-category='paragraph' style='font-size:14px'>68 KB 금쪽같은 "
 "펫보험(강아지)(무배당)(26.01)</p><p id='76' data-category='list'></p><br><p id='77' "
 "data-category='paragraph' style='font-size:14px'>\uf000 계약자는 회사가 정당한 사유 없이 "
 '제1항의 요구를 따르지 않는 경우 해당'),
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
