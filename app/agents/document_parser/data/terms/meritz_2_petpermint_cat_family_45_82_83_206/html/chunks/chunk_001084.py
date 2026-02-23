from langchain_core.documents import Document

chunk = Document(
    page_content=(". 장해의 분류</p><footer id='32' style='font-size:14px'>200</footer><table "
 "id='33' style='font-size:16px'><thead><tr><td>장해의 "
 '분류</td><td>지급률</td></tr></thead><tbody><tr><td>1) 신경계에 장해가 남아 일상생활 기본동작에 제한을 '
 '남긴 때</td><td>10∼100</td></tr><tr><td>2) 정신행동에 극심한 장해를 '
 '남긴때</td><td>100</td></tr><tr><td>3) 정신행동에 심한 장해를'),
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
