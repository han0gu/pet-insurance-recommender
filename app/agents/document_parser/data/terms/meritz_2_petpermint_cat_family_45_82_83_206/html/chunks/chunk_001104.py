from langchain_core.documents import Document

chunk = Document(
    page_content=("확<br>인되는 뇌전증 발작의 빈도 및 양상을 기준으로</p><footer id='52' "
 "style='font-size:14px'>203</footer><h1 id='53' "
 "style='font-size:20px'>한다.</h1><br><p id='54' data-category='list' "
 "style='font-size:16px'>다) “심한 뇌전증 발작”이라 함은 월 8회 이상의 중<br>증발작이 연 6개월 이상의 기간에 "
 '걸쳐 발생하<br>고, 발작할 때 유발된 호흡장애, 흡인성 폐렴,<br>심한 탈진, 구역질, 두통,'),
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
