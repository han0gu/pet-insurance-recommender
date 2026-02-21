from langchain_core.documents import Document

chunk = Document(
    page_content=('알린 내용이나 건강진단 내용이 보험금<br>지급사유의 발생에 영향을 미쳤음을 회사가 증명하는<br>경우<br>② 제17조(알릴 의무 '
 '위반의 효과)를 준용하여 회사가 보<br>장을 하지 않을 수 있는 경우<br>③ 진단계약에서 보험금 지급사유가 발생할 때까지 '
 '진단<br>을 받지 않은 경우'),
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
