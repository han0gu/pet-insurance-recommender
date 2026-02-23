from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:16px'>\uf000 회사는 제1항의 통지로 인하여 위험의 변동이 발생한 경<br>우에는 "
 "제13조(계약내용의 변경 등)에 따라 계약내용을 변<br>경할 수 있습니다.</p><footer id='57' "
 "style='font-size:14px'>91</footer><figure id='58'><img "
 'style=\'font-size:16px\' alt="[위험변경에 따른 계약변경 절차]'),
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
