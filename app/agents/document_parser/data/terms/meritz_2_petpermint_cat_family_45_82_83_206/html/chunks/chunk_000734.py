from langchain_core.documents import Document

chunk = Document(
    page_content=('. 단, 흡인, 천자 등<br>의 조치, 신경(神經)차단(NERVE BLOCK), 미용성형 목적의<br>수술, 피임목적의 수술 및 검사, '
 "진단을 위한 수술(생검,<br>복강경검사 등)은 제외합니다.</p><br><h1 id='41' "
 "style='font-size:20px'>【용어의 정의】</h1><br><p id='42' data-category='list' "
 "style='font-size:16px'>- 절단(切斷): 특정부위를 잘라 내는 것<br>- 절제(切除): 특정부위를 잘라 없애는 "
 '것<br>- 흡인(吸引): 주사기 등으로'),
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
