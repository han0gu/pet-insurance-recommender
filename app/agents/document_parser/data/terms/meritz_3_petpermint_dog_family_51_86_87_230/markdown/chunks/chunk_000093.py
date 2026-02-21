from langchain_core.documents import Document

chunk = Document(
    page_content=('그 때부터 소멸됩니다.\n'
 '\uf000 제1항에 따라 이 계약의 보장책임이 소멸된 때에는 회사\n'
 '는 이 보장책임의 해약환급금을 지급하지 않으며, 그 때까\n'
 '지「보험료 및 해약환급금 산출방법서」에서 정하는 바에\n'
 '따라 회사가 적립한 적립부분의 계약자적립액(중도인출이\n'
 '있는 경우에는 중도인출 원금과 이자를 차감하고 적립한 금\n'
 '액을 말합니다) 및 미경과보험료를 계약자에게 지급합니다.\n'
 '\uf000 피보험자가 사망한 경우에는 이 계약은 소멸되며, 이 경\n'
 '우 회사는 그 때까지「보험료 및 해약환급금 산출방법서」\n'
 '에서 정한 사망 당시 계약자적립액(중도인출이 있는 경우'),
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
