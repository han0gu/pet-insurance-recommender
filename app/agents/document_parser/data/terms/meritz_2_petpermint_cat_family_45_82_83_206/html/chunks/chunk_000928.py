from langchain_core.documents import Document

chunk = Document(
    page_content=('감았<br>을 때 각막을 완전히 덮을 수 없는 경우를 말한다.<br>11) 외상이나 화상 등으로 안구의 적출이 불가피한 경우<br>에는 '
 '외모의 추상(추한 모습)이 가산된다'),
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
