from langchain_core.documents import Document

chunk = Document(
    page_content=('을 알리지 않은 경우에는 계약자 또는 보험수익자가 회사에\n'
 '알린 최종의 주소 또는 연락처로 등기우편 등 우편물에 대\n'
 '한 기록이 남는 방법으로 회사가 알린 사항은 일반적으로\n'
 '도달에 필요한 기간이 지난 때에 계약자 또는 보험수익자에\n'
 '게 도달된 것으로 봅니다.56# 제13조(보험수익자의 지정)보험수익자를 지정하지 않은 때에는 보험수익자를 만기환급\n'
 '금의 경우는 계약자로 하고, 사망보험금의 경우는 피보험자\n'
 '의 법정상속인, 이 외의 보험금은 피보험자로 합니다.# 【법정상속인】법정상속인이라 함은 피상속인의 사망에 의하여 민법의'),
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
