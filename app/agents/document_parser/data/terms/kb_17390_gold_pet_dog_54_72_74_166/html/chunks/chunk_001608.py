from langchain_core.documents import Document

chunk = Document(
    page_content=("지연유합은</p><br><p id='119' data-category='list' "
 "style='font-size:16px'>제외한다.<br>13) ‘가관절이 남아 약간의 장해를 남긴 때’라 함은 경골과 종아리뼈 중 "
 '어<br>느 한 뼈에 가관절이 남은 경우를 말한다'),
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
