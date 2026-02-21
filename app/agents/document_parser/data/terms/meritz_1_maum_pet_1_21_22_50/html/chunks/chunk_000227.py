from langchain_core.documents import Document

chunk = Document(
    page_content=('. 그러나 계약자 또는 피보험자의 책임있는 사유로 지급이 지연된<br>때에는 그 해당기간에 대한 이자는 더하여 지급하지 '
 "않습니다.</p><h1 id='46' style='font-size:14px'>제8조(보험금 등의 지급한도)</h1><br><p "
 "id='47' data-category='paragraph' style='font-size:14px'>① 회사는 1회의 보험사고에 대하여 "
 '다음과 같이 보상합니다'),
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
