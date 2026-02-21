from langchain_core.documents import Document

chunk = Document(
    page_content=('검안서, 증명서 또는 처방전을 발급하지<br>못하며, 「약사법」 제85조제6항에 따른 동물용 의약<br>품(이하 "동물용 의약품"이라 '
 '한다)을 처방·투약하<br>지 못한다'),
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
