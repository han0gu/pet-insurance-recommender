from langchain_core.documents import Document

chunk = Document(
    page_content=('내에서 보험금을 지급하며 연간 첫 번째 의<br>료행위 이후에 발생한 MRI,CT 및 내시경처치에 대하여 제2<br>항의 연간 두번째 '
 '이상 지급한도 내에서 보험금을 지급합니<br>다'),
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
