from langchain_core.documents import Document

chunk = Document(
    page_content=('장해지급률이 상해 발생<br>일부터 180일 이내에 확정되지 않는 경우에는 상해 발생일<br>부터 180일이 되는 날의 의사 진단에 '
 '기초하여 고정될 것으<br>로 인정되는 상태를 장해지급률로 결정합니다'),
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
