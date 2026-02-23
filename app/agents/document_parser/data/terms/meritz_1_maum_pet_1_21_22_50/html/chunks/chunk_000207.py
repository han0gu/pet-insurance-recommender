from langchain_core.documents import Document

chunk = Document(
    page_content=('. 피보험자가 피해자에게 지급할 책임을 지는 법률상의 손해배상금<br>2. 계약자 또는 피보험자가 지출한 아래의 비용</p><br><p '
 "id='24' data-category='list' style='font-size:14px'>가. 피보험자가 제11조(손해방지의무)의 "
 '제1항 제1호의 손해의 방지 또는 경감을 위<br>하여 지출한 필요 또는 유익하였던 비용<br>나'),
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
