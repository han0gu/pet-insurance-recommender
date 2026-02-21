from langchain_core.documents import Document

chunk = Document(
    page_content=("id='97' style='font-size:20px'>제13조(계약내용의 변경 등)</h1><br><p id='98' "
 "data-category='paragraph' style='font-size:16px'>\uf000 계약자는 회사의 승낙을 얻어 다음의 "
 '사항을 변경할 수<br>있습니다'),
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
