from langchain_core.documents import Document

chunk = Document(
    page_content=(". 이 때 적용되는 이율을「보험계약대</p><footer id='105' "
 "style='font-size:14px'>66</footer><p id='0' data-category='paragraph' "
 "style='font-size:16px'>출이율」이라 하며, 회사에서 별도로 정한 방법에 따라<br>결정합니다"),
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
