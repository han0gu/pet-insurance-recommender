from langchain_core.documents import Document

chunk = Document(
    page_content=("합니</p><br><p id='32' data-category='paragraph' "
 "style='font-size:14px'>다.</p><br><table id='33' "
 "style='font-size:14px'><thead></thead><tbody><tr><td>용 어 풀 "
 '이</td><td>보장개시일</td></tr><tr><td colspan="2">회사가 보장을 개시하는 날로서 계약이 성립되고 제1회 '
 '보험료를 받은 날을 말 하나, 회사가 승낙하기 전이라도 청약과 함께 제1회 보험료를 받은 경우에는 제1회 보험료를 받은 날을'),
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
