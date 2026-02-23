from langchain_core.documents import Document

chunk = Document(
    page_content=('비<br>용, 상담 수수료, 지도 비용 및 이와 동종의 비용<br>⑮ 왕진 비용, 가입동물의 이송비, 동물병원에 가지 않<br>고 약제만 '
 '배달되는 배달료 및 이와 동종의 비용<br>⑯ 안락사 비용, 시체처치 및 해부검사, 장례비, 이장비<br>등 사후에 필요한 비용<br>⑰ '
 '마이크로 칩 이식 비용, 각종 증빙서류의 작성비용<br>(우송비 포함)<br>⑱ 과잉진료행위로 인한 비용</p><footer '
 "id='33' style='font-size:14px'>140</footer><p id='34'"),
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
