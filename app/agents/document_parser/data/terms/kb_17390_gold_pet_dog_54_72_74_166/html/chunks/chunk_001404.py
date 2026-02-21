from langchain_core.documents import Document

chunk = Document(
    page_content=("정하지 않은 사항은 보통약관 및 해당 특별약관을 따릅니다.</p><p id='64' "
 "data-category='list'></p><br><p id='65' data-category='paragraph' "
 "style='font-size:20px'>6"),
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
