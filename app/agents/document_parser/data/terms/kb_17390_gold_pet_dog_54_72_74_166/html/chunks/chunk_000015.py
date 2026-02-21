from langchain_core.documents import Document

chunk = Document(
    page_content=(". 기간과 날짜 관련</p><br><p id='9' data-category='paragraph' "
 "style='font-size:16px'>용어</p><table id='10' "
 "style='font-size:16px'><thead><tr><td>용 어</td><td>정 "
 '의</td></tr></thead><tbody><tr><td>보험기간</td><td>계약에 따라 보장을 받는 기간을 말합니다'),
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
