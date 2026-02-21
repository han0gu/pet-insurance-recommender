from langchain_core.documents import Document

chunk = Document(
    page_content=('해당 관절의 운동범위 합계가 정상운동범위의<br>1/2 이하로 제한된 경우<br>나) 근전도 검사상 불완전한 '
 '손상(incomplete<br>injury) 소견이 있으면서 도수근력검사(MMT)<br>에서 근력이 2등급(poor)인 '
 "경우</p><br><p id='39' data-category='paragraph' style='font-size:20px'>10) "
 "“관절 하나의 기능에 약간의 장해를 남긴 때”라<br>함은 아래의 경우 중 하나에 해당하는 때를 말한다.</p><br><p id='40' "
 "data-category='list'"),
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
