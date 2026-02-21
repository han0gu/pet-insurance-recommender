from langchain_core.documents import Document

chunk = Document(
    page_content=("제1조(보험금의 지급사유) 제1항의</p><br><p id='32' data-category='paragraph' "
 "style='font-size:14px'>2회 이상 입원한 경우 이를 1회 입원으로 보아 각 입원일수를 더합니다"),
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
