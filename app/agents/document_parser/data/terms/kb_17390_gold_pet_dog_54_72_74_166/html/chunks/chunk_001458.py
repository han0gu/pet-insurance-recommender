from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:18px'>- 139 -</p><br><p id='130' data-category='paragraph' "
 "style='font-size:16px'>제4조(준용규정)</p><br><p id='131' "
 "data-category='paragraph' style='font-size:16px'>이 특별약관에 정하지 아니한 사항에 대하여는 "
 "보통약관 및 해당 특별약관의 규정을</p><br><p id='132' data-category='paragraph'"),
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
