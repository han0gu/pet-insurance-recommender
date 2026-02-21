from langchain_core.documents import Document

chunk = Document(
    page_content=("보험료)</p><br><p id='232' data-category='paragraph' style='font-size:16px'>이 "
 "특별약관의 보험료는</p><br><p id='233' data-category='paragraph' "
 "style='font-size:16px'>없습니다.</p><h1 id='234' "
 "style='font-size:16px'>제9조(준용규정)</h1><br><p id='235' "
 "data-category='paragraph' style='font-size:14px'>제</p><p id='236'"),
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
