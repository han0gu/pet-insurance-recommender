from langchain_core.documents import Document

chunk = Document(
    page_content=('. 단, 계약 청약일 현재 부담보 기간을「계약의 보<br>험기간」으로 적용한 유사계약이 유지중이거나, 계약 청약<br>일 전 6개월 '
 '이내에 계약자 및 피보험자의 요구 또는 보험<br>료 납입 연체로 해지된 경우「계약의 보험기간」이내에서<br>계약의 부담보 기간을 '
 '적용하며, 유사계약 청약일 이후 제1<br>항에서 정한 질병과 관련한 새로운 위험(재진단·치료 등은<br>해당하지 않습니다)이 '
 '발생하거나, 새로운 질병에 대한 보<br>장이 추가(입원비, 수술비, 진단비 등 보장 범위의 변경 또<br>는 확대는 해당하지 '
 '않습니다)된 경우'),
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
