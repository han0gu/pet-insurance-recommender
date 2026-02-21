from langchain_core.documents import Document

chunk = Document(
    page_content=('고형식(밥,<br>빵 등)만 섭취 가능한 경우<br>나) 위‧아래턱(상ㆍ하악)의 가운데 앞니(중절치)간<br>최대 개구(입을 벌림)운동이 '
 '2cm이하로 제한되<br>는 경우<br>다) 위‧아래턱(상ㆍ하악)의 부정교합(전방, 측방)이<br>1cm이상인 경우<br>라) 양측 각 '
 '1개 또는 편측 2개 이하의 치아만 교합<br>되는 상태<br>마) 연하기능검사(비디오 투시검사)상 연하장애가<br>있고, 유동식 섭취시 '
 "간헐적으로 흡인이 발생<br>하고 부드러운 고형식 외에는 섭취가 불가능한<br>상태</p><br><p id='44'"),
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
