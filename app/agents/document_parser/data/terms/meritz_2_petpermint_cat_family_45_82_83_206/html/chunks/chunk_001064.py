from langchain_core.documents import Document

chunk = Document(
    page_content=('. 단, 제거가 불가능한<br>경우에는 고정물 등이 있는 상태에서 장해를 평가한<br>다.<br>2) 관절을 사용하지 않아 발생한 '
 "일시적인 기능장해(예를<br>들면 캐스트로 환부를 고정시켰기 때문에 치유 후의 관</p><footer id='8' "
 "style='font-size:14px'>197</footer><p id='9' data-category='paragraph' "
 "style='font-size:18px'>절에 기능장해가 발생한 경우)는 장해로 평가하지 않<br>는다.</p><br><p id='10'"),
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
