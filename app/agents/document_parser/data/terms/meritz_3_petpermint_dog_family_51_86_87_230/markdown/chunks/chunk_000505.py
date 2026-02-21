from langchain_core.documents import Document

chunk = Document(
    page_content=('계를 같이 하는 가족에 대한 것인 경우에는 그 권리를 취득\n'
 '하지 못합니다. 다만, 손해가 그 가족의 고의로 인하여 발\n'
 '생한 경우에는 그 권리를 취득합니다.# 제12조(계약 후 알릴 의무)\uf000 계약을 맺은 후 아래와 같은 사실이 생긴 경우에는 계약\n'
 '자 또는 피보험자는 지체없이 서면으로 회사에 알리고 보험\n'
 '증권에 확인을 받아야 합니다.- ① 청약서의 기재사항을 변경하고자 할 때 또는 변경이\n'
 '- 생겼음을 알았을 때\n'
 '- ② 이 계약에서 보장하는 위험과 동일한 위험을 보장하는\n'
 '- 계약을 다른 보험자와 맺으려고 하든지 또는 이와 같'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
