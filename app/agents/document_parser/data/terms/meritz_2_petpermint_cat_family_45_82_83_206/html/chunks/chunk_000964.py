from langchain_core.documents import Document

chunk = Document(
    page_content=("id='49' style='font-size:20px'>장해의 대상이 되지 않으나, 선천적으로 영구치 결<br>손이 있는 경우에는 유치의 "
 "결손을 후유장해로 평가<br>한다.</h1><br><p id='50' data-category='paragraph' "
 "style='font-size:16px'>16) 가철성 보철물(신체의 일부에 붙였다 떼었다 할 수<br>있는 틀니 등)의 파손은 "
 "후유장해의 대상이 되지 않<br>는다.</p><h1 id='51' style='font-size:20px'>5"),
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
