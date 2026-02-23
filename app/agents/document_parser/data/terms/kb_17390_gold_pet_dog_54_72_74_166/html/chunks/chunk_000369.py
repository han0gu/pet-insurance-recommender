from langchain_core.documents import Document

chunk = Document(
    page_content=('장해분류표에 해당되지 않는 후유장해는 피보험자의 직업, 연령, 신분 또는 성별<br>등에 관계없이 신체의 장해정도에 따라 장해분류표의 '
 '구분에 준하여 지급액을 결<br>정합니다'),
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
