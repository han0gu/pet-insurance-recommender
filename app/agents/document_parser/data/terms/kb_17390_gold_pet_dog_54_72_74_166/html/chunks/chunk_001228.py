from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 의무보험인 경우에는 회사의 서면동의가 없는 경우에도 청약서에 기재된<br>사업을 양도하였을 때 계약으로 인하여 생긴 권리와 '
 "의무를 함께 양도 한 것으로 봅<br>니다.</p><h1 id='24' style='font-size:16px'>제20조(사기에 "
 "의한</h1><br><p id='25' data-category='paragraph' "
 "style='font-size:16px'>계약)</p><br><p id='26' data-category='list' "
 "style='font-size:16px'>\uf000 계약자, 피보험자 또는"),
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
