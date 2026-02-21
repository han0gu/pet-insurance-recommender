from langchain_core.documents import Document

chunk = Document(
    page_content=('사고가<br>그 증상을 악화시킨 부분만큼, 즉 이 사고와의 관여<br>도를 산정하여 평가한다.<br>4) 추간판탈출증으로 인한 신경 '
 '장해는 수술 또는 시술(비<br>수술적 치료) 후 6개월 이상 지난 후에 평가한다.<br>5) 신경학적 검사상 나타난 저린감이나 방사통 '
 '등 신경자극<br>증상의 원인으로 CT, MRI 등 영상검사에서 추간판탈출증<br>이 확인된 경우를 추간판탈출증으로 진단하며, 수술 '
 '여<br>부에 관계없이 운동장해 및 기형장해로 평가하지 않는<br>다.<br>6) 심한 운동장해란 다음 중 어느 하나에 해당하는'),
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
