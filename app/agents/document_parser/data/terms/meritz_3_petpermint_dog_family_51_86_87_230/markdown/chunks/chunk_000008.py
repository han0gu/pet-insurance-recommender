from langchain_core.documents import Document

chunk = Document(
    page_content=('52| 용어 | 정의 |\n'
 '| --- | --- |\n'
 '| 계약자 적립액 | 장래의 해약환급금 등을 지급하기 위하여 계 약자가 납입한 보험료 중 일정액을 기준으로 보험료 및 해약환급금 '
 '산출방법서에서 정한 방법에 따라 계산한 금액을 말합니다. |\n'
 '| 해약 환급금 | 계약이 해지되는 때에 회사가 계약자에게 돌 려주는 금액을 말합니다. |\n'
 '# 【연단위 복리 】회사가 지급할 금전에 이자를 줄 때, 1년마다 마지막\n'
 '날에 그 이자를 원금에 더한 금액을 다음 1년의 원금으\n'
 '로 하는 이자 계산방법을 말합니다.'),
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
