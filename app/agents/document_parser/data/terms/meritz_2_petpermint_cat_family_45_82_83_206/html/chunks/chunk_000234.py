from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:20px'>【보험연도】</h1><br><p id='19' data-category='paragraph' "
 "style='font-size:16px'>당해 연도 보험계약 해당일부터 차년도 보험계약 해당일<br>전일까지 매1년 단위의 연도를 "
 '말합니다'),
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
