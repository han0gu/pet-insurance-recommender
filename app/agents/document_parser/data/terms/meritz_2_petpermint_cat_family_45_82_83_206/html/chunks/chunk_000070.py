from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:16px'>\uf000 계약자(보험수익자가 계약자와 다른 경우 보험수익자를<br>포함합니다)는 주소 또는 "
 '연락처가 변경된 경우에는 지체없<br>이 그 변경내용을 회사에 알려야 합니다.<br>\uf000 제1항에서 정한대로 계약자 또는 '
 '보험수익자가 변경내용<br>을 알리지 않은 경우에는 계약자 또는 보험수익자가 회사에<br>알린 최종의 주소 또는 연락처로 등기우편 등 '
 '우편물에 대<br>한 기록이 남는 방법으로 회사가 알린 사항은 일반적으로<br>도달에 필요한 기간이 지난 때에 계약자 또는 '
 '보험수익자에<br>게 도달된'),
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
