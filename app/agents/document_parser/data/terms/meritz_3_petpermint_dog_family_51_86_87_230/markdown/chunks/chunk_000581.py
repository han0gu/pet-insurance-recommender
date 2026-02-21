from langchain_core.documents import Document

chunk = Document(
    page_content=('| 만기환급금(보통약관 제10조 제1항) 및 해약환급금 (보통약관 제35조 제1항) (특별약관이 부가된 경우 특별약관의 해약환급금 포함) '
 '| 청구일의 다음날부터 지급일까지의 기간 | 보험계약대출이율 |\n'
 '주) 1. 회사가 만기환급금의 지급시기 도래 7일 이전에\n'
 '지급 사유와 금액을 알리지 않은 경우, 지급사\n'
 '유가 발생한 날의 다음 날부터 청구일까지의 기\n'
 '간은 [보장]공시이율을 적용하여 계산한 이자를\n'
 '지급합니다.- 2. 지급이자의 계산은 연단위 복리로 계산하며, 일\n'
 '- 자 계산합니다.'),
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
