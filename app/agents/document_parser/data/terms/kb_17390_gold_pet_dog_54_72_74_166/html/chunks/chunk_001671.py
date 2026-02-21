from langchain_core.documents import Document

chunk = Document(
    page_content=('조절되지 않는 뇌전증을 말하며, 진료기록에 기재되어 객관적<br>으로 확인되는 뇌전증 발작의 빈도 및 양상을 기준으로 '
 "한다.</p><br><p id='15' data-category='list'></p><br><p id='16' "
 "data-category='paragraph' style='font-size:20px'>- 154 -</p><p id='17' "
 "data-category='list'></p><p id='18' data-category='list' "
 "style='font-size:14px'>다) “심한 뇌전증 발작”이라 함은"),
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
