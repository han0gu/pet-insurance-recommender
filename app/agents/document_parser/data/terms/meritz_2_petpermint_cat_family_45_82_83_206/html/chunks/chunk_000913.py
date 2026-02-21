from langchain_core.documents import Document

chunk = Document(
    page_content='동일한 신체부위에 2가지 이상의 장해가 발생한 경우에<br>는 합산하지 않고 그 중 높은 지급률을 적용함을 원칙<br>으로 한다',
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
