from langchain_core.documents import Document

chunk = Document(
    page_content=("부드러운 고형식 외에는 섭취가 불가능한<br>상태</p><br><p id='44' data-category='paragraph' "
 "style='font-size:16px'>5) 개구(입을 벌림)장해는 턱관절의 이상으로 개구(입<br>을 벌림)운동 제한이 있는 상태를 "
 "말하며, 최대 개</p><footer id='45' style='font-size:14px'>182</footer><p id='46' "
 "data-category='paragraph' style='font-size:18px'>구(입을 벌림)상태에서 위‧아래턱(상ㆍ하악)의"),
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
