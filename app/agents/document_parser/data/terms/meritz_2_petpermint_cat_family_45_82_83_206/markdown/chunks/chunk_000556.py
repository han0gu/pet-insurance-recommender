from langchain_core.documents import Document

chunk = Document(
    page_content=('- 상의 척추체(척추뼈 몸통)를 유합(아물어 붙음)\n'
 '- 또는 고정한 상태\n'
 '- 나) 머리뼈(두개골), 제1경추, 제2경추를 모두 유합\n'
 '186# 또는 고정한 상태7) 뚜렷한 운동장해란 다음 중 어느 하나에 해당하는 경우\n'
 '를 말한다.- 가) 척추체(척추뼈 몸통)에 골절 또는 탈구로 3개의\n'
 '- 척추체(척추뼈 몸통)를 유합(아물어 붙음) 또는\n'
 '- 고정한 상태\n'
 '- 나) 머리뼈(두개골)와 제1경추 또는 제1경추와 제2경\n'
 '- 추를 유합 또는 고정한 상태\n'
 '- 다) 머리뼈(두개골)와 상위목뼈(상위경추: 제1, 2경'),
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
