from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:14px'>물</p><h1 id='143' style='font-size:16px'>\uf000 이 "
 "특별약관에서</h1><br><p id='144' data-category='paragraph' "
 "style='font-size:16px'>정하지 않은 사항은 반려동물(강아지) 일반조항을 따릅니다"),
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
