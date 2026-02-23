from langchain_core.documents import Document

chunk = Document(
    page_content=("손상으</h1><br><p id='186' data-category='paragraph' "
 "style='font-size:16px'>법<br>로 “<붙임>일상생활 기본동작(ADLs) 제한 장해평가표”의 5가지 기 "
 'ㆍ<br>본동작중 하나 이상의 동작이 제한되었을 때를 말한다'),
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
