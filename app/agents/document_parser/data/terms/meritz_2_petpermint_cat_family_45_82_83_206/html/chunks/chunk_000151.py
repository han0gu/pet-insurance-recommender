from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만,<br>심신박약자가 계약을 체결하거나 소속 단체의 규약에<br>따라 단체보험의 피보험자가 될 때에 의사능력이 있는<br>경우에는 '
 '계약이 유효합니다.<br>③ 계약을 체결할 때 계약에서 정한 피보험자의 나이에<br>미달되었거나 초과되었을 경우'),
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
