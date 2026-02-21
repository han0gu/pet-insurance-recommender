from langchain_core.documents import Document

chunk = Document(
    page_content=('부활(효력회복))를 따릅니다.<br>이 경우 부활(효력회복)일을 계약일로 하여 제3항 및 제4항<br>의 보장개시일을 '
 "적용합니다.</p><footer id='30' style='font-size:14px'>129</footer><p id='31' "
 "data-category='paragraph' style='font-size:20px'>제2조(보험금을 지급하지 않는 "
 "사유)</p><br><p id='32' data-category='paragraph' "
 "style='font-size:20px'>\uf000 회사는 다음 중 어느 한 가지로 보험금"),
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
