from langchain_core.documents import Document

chunk = Document(
    page_content=('청구에 따라 이미 확정된 보험금을 먼저 가지급<br>합니다.<br>\uf000 제2항에 따라 추가적인 조사가 이루어지는 경우, '
 '회사는<br>보험수익자의 청구에 따라 회사가 추정하는 보험금의 50%<br>상당액을 가지급보험금으로 지급합니다.</p><br><h1 '
 "id='65' style='font-size:20px'>【가지급보험금】</h1><br><p id='66' "
 "data-category='paragraph' style='font-size:16px'>보험금이 지급기한 내에 지급되지 못할 것으로 "
 '판단되는 경우 회<br>사가 예상되는 보험금의'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
