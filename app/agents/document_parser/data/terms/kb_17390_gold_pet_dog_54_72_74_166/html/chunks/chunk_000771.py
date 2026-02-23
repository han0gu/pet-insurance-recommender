from langchain_core.documents import Document

chunk = Document(
    page_content=('따라 보장을 받는 기간을 말합니다.</td></tr></thead><tbody><tr><td></td><td>회사가 영업점에서 정상적으로 '
 '영업하는 날을 말하며, 토</td></tr><tr><td>영업일</td><td>요일, "관공서의 공휴일에 관한 규정"에 따른 공휴일과 노 '
 "동절을 제외합니다.</td></tr></tbody></table><br><table id='125' "
 "style='font-size:16px'><thead></thead><tbody><tr><td></td><td>(대통령령"),
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
