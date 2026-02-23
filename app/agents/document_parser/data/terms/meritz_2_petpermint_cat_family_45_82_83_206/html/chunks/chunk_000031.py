from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 피보험자<br>가 심신상실 등으로 자유로운 의사결정을 할 수 없는<br>상태에서 자신을 해친 경우에는 보험금을 '
 "지급합니다.</p><br><h1 id='40' style='font-size:20px'>【심신상실】</h1><br><p id='41' "
 "data-category='paragraph' style='font-size:16px'>정신병, 정신박약, 심한 의식장애 등의 심신장애로 "
 "인하<br>여 사물 변별 능력 또는 의사 결정 능력이 없는 상태를<br>말합니다.</p><br><p id='42'"),
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
