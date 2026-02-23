from langchain_core.documents import Document

chunk = Document(
    page_content=(". 단, 수술에서 아래에 정한 사항은 제외합니다.<br>1. 흡인(吸引)</p><br><p id='41' "
 "data-category='list' style='font-size:14px'>2. 천자(穿刺) 등의 조치<br>3. 미용성형 목적의 "
 '수술<br>4'),
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
