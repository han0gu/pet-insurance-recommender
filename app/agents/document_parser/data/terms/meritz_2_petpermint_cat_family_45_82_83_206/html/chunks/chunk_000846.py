from langchain_core.documents import Document

chunk = Document(
    page_content=('. 보상하지 않는 질병(부담보 질병)은 회사가 정한 기준에<br>따라 직접 관련이 있는 특정질병으로 제한합니다.</p><br><table '
 "id='12' style='font-size:14px'><thead><tr><td>구 "
 '분</td><td>특정질병</td><td>분류코드</td><td>항목명</td></tr></thead><tbody><tr><td '
 'rowspan="28">1</td><td rowspan="28">뒷다리 근골격계 질환</td><td>AEB003</td><td>뒷다리의'),
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
