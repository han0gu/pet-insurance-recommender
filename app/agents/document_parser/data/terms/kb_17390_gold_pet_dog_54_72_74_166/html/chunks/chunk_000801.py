from langchain_core.documents import Document

chunk = Document(
    page_content=("id='166' data-category='paragraph' style='font-size:14px'>청구)에서 정한 서류를 접수한 "
 "때에는 접수증을 드리고</p><p id='167' data-category='paragraph' "
 "style='font-size:18px'>- 102 -</p><br><p id='168' data-category='paragraph' "
 "style='font-size:14px'>휴대전화 문자메시지 또는 전자우편 등으로도 송부하며, 그 서류를 접수한 날부<br>터 3영업일 "
 '이내에 보험금을'),
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
