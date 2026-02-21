from langchain_core.documents import Document

chunk = Document(
    page_content=('. \uf000 제2항에서 "그와 유사한 구조로 되어 있는 자동차"는 다음 각 호에 해당하는 '
 "자동</td></tr></tbody></table><br><p id='188' data-category='paragraph' "
 "style='font-size:14px'>차를 포함합니다.<br>1"),
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
