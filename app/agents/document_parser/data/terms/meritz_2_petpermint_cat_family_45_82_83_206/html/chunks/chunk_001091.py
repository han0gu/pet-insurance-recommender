from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:20px'>가) 정신행동장해는 보험기간중에 발생한 뇌의 질병</p><footer id='39' "
 "style='font-size:14px'>201</footer><p id='40' data-category='paragraph' "
 "style='font-size:16px'>또는 상해를 입은 후 18개월이 지난 후에 판정<br>함을 원칙으로 한다"),
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
