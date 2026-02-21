from langchain_core.documents import Document

chunk = Document(
    page_content=("250만원</p><br><p id='9' data-category='paragraph' "
 "style='font-size:16px'>\uf000 제1항에도 불구하고 보장개시일로부터 그 날을 포함하여<br>30일 이내에 발생한 "
 '질병은 보상하지 않습니다'),
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
