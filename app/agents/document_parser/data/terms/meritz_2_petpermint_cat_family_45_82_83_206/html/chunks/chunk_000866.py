from langchain_core.documents import Document

chunk = Document(
    page_content=("id='16' style='font-size:14px'><thead><tr><td>구 "
 '분</td><td>특정질병</td><td>분류코드</td><td>항목명</td></tr></thead><tbody><tr><td '
 'rowspan="21"></td><td rowspan="21">피부질환</td><td>AGA004</td><td>기타 비뇨기계 양성 '
 '신생물</td></tr><tr><td>AGB004</td><td>기타 비뇨기계 악성 '
 '신생물</td></tr><tr><td>AGC004</td><td>기타 비뇨기계 신생물 (양성 또는 악성이'),
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
