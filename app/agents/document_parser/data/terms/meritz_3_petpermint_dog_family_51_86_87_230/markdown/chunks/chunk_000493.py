from langchain_core.documents import Document

chunk = Document(
    page_content=('로 가입하여야 하는 보험으로서 공제계약을 포함합니다.\n'
 '\uf000 피보험자가 의무보험에 가입하여야 함에도 불구하고 가\n'
 '입하지 않은 경우에는 그가 가입했더라면 의무보험에서 보\n'
 '상했을 금액을 제1항의 의무보험에서 보상하는 금액으로 봅\n'
 '니다.# 제7조(보험금의 분담)\uf000 회사는 이 특별약관에서 보장하는 위험과 같은 위험을\n'
 '보장하는 다른 계약(공제계약을 포함합니다)이 있을 경우\n'
 '각 계약에 대하여 다른 계약이 없는 것으로 하여 각각 산출\n'
 '한 보상책임액의 합계액이 손해액을 초과할 때에는 아래에\n'
 '따라 손해를 보상합니다. 이 특별약관과 다른 계약이 모두'),
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
