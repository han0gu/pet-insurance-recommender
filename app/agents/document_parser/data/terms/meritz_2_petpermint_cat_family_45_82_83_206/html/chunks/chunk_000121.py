from langchain_core.documents import Document

chunk = Document(
    page_content=(". 다만, 회사는 계약자<br>가 제1회 보험료를 신용카드로 납입한 계약의 승낙을 거절</p><footer id='70' "
 "style='font-size:14px'>63</footer><p id='71' data-category='paragraph' "
 "style='font-size:16px'>하는 경우에는 신용카드의 매출을 취소하며 이자를 더하여<br>지급하지 "
 '않습니다.<br>\uf000 회사가 제2항에 따라 일부보장 제외 조건을 붙여 승낙하<br>였더라도 청약일로부터 5년(갱신형 계약의 '
 '경우에는 최초<br>계약의 청약일 이후 5년)이 지나는'),
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
