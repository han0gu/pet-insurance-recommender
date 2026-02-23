from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:14px'>188</footer><h1 id='18' style='font-size:20px'>하나에 "
 "해당하는 때를 말한다.</h1><br><p id='19' data-category='list' "
 "style='font-size:16px'>가) 천장관절 또는 치골문합부가 분리된 상태로 치유<br>되었거나 좌골이 2.5cm 이상 분리된 "
 '부정유합 상태<br>나) 육안으로 변형(결손을 포함)을 명백하게 알 수<br>있을 정도로 방사선 검사로 측정한 각(角) 변형<br>이 '
 '20° 이상인 경우<br>다) 미골의'),
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
