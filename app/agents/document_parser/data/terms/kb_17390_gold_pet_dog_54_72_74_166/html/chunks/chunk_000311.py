from langchain_core.documents import Document

chunk = Document(
    page_content=("id='158' data-category='paragraph' "
 "style='font-size:14px'>제45조(소멸시효)</p><br><table id='159' "
 "style='font-size:14px'><thead></thead><tbody><tr><td>적립액 반환청구권은 "
 '3년간</td><td>행사하지 않으면 소멸시효가 완성됩니다.</td></tr><tr><td colspan="2">부 가 설 명 소멸시효 '
 '소멸시효는 해당 청구권을 행사할 수 있는 때부터 진행합니다'),
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
