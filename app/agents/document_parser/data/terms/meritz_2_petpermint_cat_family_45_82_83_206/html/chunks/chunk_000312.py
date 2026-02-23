from langchain_core.documents import Document

chunk = Document(
    page_content=("전 알릴 의무】</h1><br><p id='48' data-category='paragraph' "
 "style='font-size:20px'>상법 제651조(고지의무위반으로 인한 계약해지)에서 정<br>하고 있는 의무"),
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
