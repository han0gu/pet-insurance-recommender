from langchain_core.documents import Document

chunk = Document(
    page_content=(". 다</p><p id='155' data-category='paragraph' style='font-size:14px'>70 KB "
 "금쪽같은 펫보험(강아지)(무배당)(26.01)</p><br><h1 id='156' style='font-size:14px'>만, 회사와 "
 "계약자가</h1><br><h1 id='157' style='font-size:14px'>합의하여 관할법원을 달리 정할 수 "
 "있습니다.</h1><br><p id='158' data-category='paragraph'"),
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
