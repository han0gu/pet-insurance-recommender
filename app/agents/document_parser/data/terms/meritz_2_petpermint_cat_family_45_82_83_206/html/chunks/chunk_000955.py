from langchain_core.documents import Document

chunk = Document(
    page_content=("상태</p><br><p id='42' data-category='paragraph' style='font-size:20px'>4) "
 '“씹어먹는 기능에 약간의 장해를 남긴 때“라 함은<br>아래의 경우 중 하나 이상에 해당되는 때를 말한다.</p><br><p '
 "id='43' data-category='list' style='font-size:16px'>가) 약간의 개구(입을 벌림)운동 제한 또는 "
 '약간의<br>저작(씹기)운동 제한으로 부드러운 고형식(밥,<br>빵 등)만 섭취 가능한 경우<br>나) 위‧아래턱(상ㆍ하악)의 가운데'),
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
