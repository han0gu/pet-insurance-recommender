from langchain_core.documents import Document

chunk = Document(
    page_content=(". 이 경우 승낙을 서면<br>등으로 알리거나 보험증권의 뒷면에 기재하여 드립니다.</p><br><p id='32' "
 "data-category='list' style='font-size:14px'>1. 보험종목<br>2. 보험기간<br>3. 보험료 "
 '납입주기, 납입방법 및 납입기간<br>4. 계약자, 피보험자 중 일부<br>5'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
