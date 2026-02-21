from langchain_core.documents import Document

chunk = Document(
    page_content=("회사는 다음 중 어느 한 가지로 보험금 지급사유가 발생<br>한 때에는 보험금을 지급하지 않습니다.</p><br><p id='39' "
 "data-category='paragraph' style='font-size:20px'>① 피보험자가 고의로 자신을 해친 경우"),
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
