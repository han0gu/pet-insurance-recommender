from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:14px'>186</footer><h1 id='0' style='font-size:20px'>또는 고정한 "
 "상태</h1><br><p id='1' data-category='paragraph' style='font-size:20px'>7) 뚜렷한 "
 "운동장해란 다음 중 어느 하나에 해당하는 경우<br>를 말한다.</p><br><p id='2' data-category='list' "
 "style='font-size:16px'>가) 척추체(척추뼈 몸통)에 골절 또는 탈구로 3개의<br>척추체(척추뼈 몸통)를"),
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
