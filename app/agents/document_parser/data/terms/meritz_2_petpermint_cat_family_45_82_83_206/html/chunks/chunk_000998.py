from langchain_core.documents import Document

chunk = Document(
    page_content=('휘어지는<br>증상) 변형이 있을 때<br>나) 척추체(척추뼈 몸통) 한 개의 압박률이 20%이상<br>인 경우 또는 한 운동단위 내에 '
 '두 개 이상 척추<br>체(척추뼈 몸통)의 압박골절로 각 척추체(척추뼈<br>몸통)의 압박률의 합이 40% 이상일 때</p><br><p '
 "id='11' data-category='list' style='font-size:20px'>12) “추간판탈출증으로 인한 심한 신경 "
 '장해”란 추간<br>판탈출증으로 추간판을 2마디이상(또는 1마디 추간판<br>에 대해 2회 이상) 수술하고도 마미신경증후군이'),
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
