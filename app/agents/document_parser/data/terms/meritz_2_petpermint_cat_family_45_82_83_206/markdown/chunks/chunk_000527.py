from langchain_core.documents import Document

chunk = Document(
    page_content=('- 평균순음역치가 80dB이상인 경우에 해당되어, 귀에다\n'
 '- 대고 말하지 않고는 큰소리를 알아듣지 못하는 경우\n'
 '- 를 말한다.\n'
 '- 4) “약간의 장해를 남긴 때”라 함은 순음청력검사 결과\n'
 '179평균순음역치가 70dB이상인 경우에 해당되어, 50cm\n'
 '이상의 거리에서는 보통의 말소리를 알아듣지 못하\n'
 '는 경우를 말한다.5) 순음청력검사를 실시하기 곤란하거나(청력의 감소가 의\n'
 '심되지만 의사소통이 되지 않는 경우, 만 3세 미만의\n'
 '소아 포함) 검사결과에 대한 검증이 필요한 경우에는\n'
 '“언어청력검사, 임피던스 청력검사, 청성뇌간반응검'),
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
