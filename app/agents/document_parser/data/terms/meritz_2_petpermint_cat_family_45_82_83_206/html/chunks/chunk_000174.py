from langchain_core.documents import Document

chunk = Document(
    page_content=('정하는 바에<br>따라 회사가 적립한 적립부분의 계약자적립액(중도인출이<br>있는 경우에는 중도인출 원금과 이자를 차감하고 적립한 '
 '금<br>액을 말합니다) 및 미경과보험료를 계약자에게 지급합니다.<br>\uf000 피보험자가 사망한 경우에는 이 계약은 소멸되며, 이 '
 '경<br>우 회사는 그 때까지「보험료 및 해약환급금 산출방법서」<br>에서 정한 사망 당시 계약자적립액(중도인출이 있는 '
 '경우<br>중도인출 원금과 이자를 차감하고 적립한 금액을 말합니다)<br>및 미경과보험료를 계약자에게 지급합니다.</p><br><h1 '
 "id='36'"),
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
