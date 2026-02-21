from langchain_core.documents import Document

chunk = Document(
    page_content=('. 단, 제거가 불가능한 경<br>우에는 고정물 등이 있는 상태에서 장해를 평가한다.<br>2) 관절을 사용하지 않아 발생한 일시적인 '
 '기능장해(예<br>를 들면 캐스트로 환부를 고정시켰기 때문에 치유후<br>의 관절에 기능장해가 발생한 경우)는 장해로 평가하<br>지 '
 '않는다.<br>3) “팔”이라 함은 어깨관절(견관절)부터 손목관절(완<br>관절)까지를 말한다.<br>4) “팔의 3대관절”이라 함은 '
 '어깨관절(견관절), 팔꿈치<br>관절(주관절), 손목관절(완관절)을 말한다.<br>5) “한팔의 손목이상을 잃었을 때”라 함은'),
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
