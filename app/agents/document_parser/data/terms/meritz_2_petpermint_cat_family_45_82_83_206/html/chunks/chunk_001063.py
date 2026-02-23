from langchain_core.documents import Document

chunk = Document(
    page_content=(". 장해판정기준</h1><br><p id='7' data-category='list' style='font-size:16px'>1) "
 '골절부에 금속내고정물 등을 사용하였기 때문에 그것<br>이 기능장해의 원인이 되는 때에는 그 내고정물 등이<br>제거된 후에 장해를 '
 '평가한다'),
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
