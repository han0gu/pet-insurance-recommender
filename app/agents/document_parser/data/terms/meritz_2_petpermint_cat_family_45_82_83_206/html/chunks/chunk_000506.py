from langchain_core.documents import Document

chunk = Document(
    page_content=('기재된 자기부담금</td><td>1일당 10만원</td></tr><tr><td>입원 중 수술을 한 날의 경우</td><td>수술당일에 '
 "한하여 1일당 150만원</td></tr></tbody></table><footer id='19' "
 "style='font-size:14px'>118</footer><p id='20' data-category='paragraph' "
 "style='font-size:20px'>【보험금 지급금액 산출방식】</p><br><p id='21' "
 "data-category='paragraph'"),
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
