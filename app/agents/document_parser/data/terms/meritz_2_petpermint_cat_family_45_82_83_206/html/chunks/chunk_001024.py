from langchain_core.documents import Document

chunk = Document(
    page_content=("상태를 말하며, 골유<br>합이 지연되는 지연유합은 제외한다.</p><br><p id='43' data-category='list' "
 "style='font-size:16px'>12) “가관절이 남아 약간의 장해를 남긴 때”라 함은<br>요골과 척골중 어느 한 뼈에 "
 '가관절이 남은 경우를<br>말한다.<br>13) “뼈에 기형을 남긴 때”라 함은 상완골 또는 요골<br>과 척골에 변형이 남아 정상에 '
 "비해 부정유합된 각<br>변형이 15° 이상인 경우를 말한다.</p><h1 id='44' style='font-size:20px'>다"),
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
