from langchain_core.documents import Document

chunk = Document(
    page_content=('입을 연체하여 해지된 계약의 부활(효력회복))를 따릅니다.\n'
 '이 경우 부활(효력회복)일을 계약일로 하여 제5항 및 제6항\n'
 '의 보장개시일을 적용합니다.# 제2조(보험금을 지급하지 않는 사유)\uf000 회사는 다음 중 어느 한 가지로 보험금 지급사유가 발생\n'
 '한 때에는 보험금을 지급하지 않습니다.- \n'
 '- ① 계약자, 피보험자, 이들의 가족 또는 사용인의 고의\n'
 '- 또는 중대한 과실\n'
 '- ② 지진, 분화, 해일, 홍수 또는 이와 유사한 자연재해로\n'
 '- 생긴 손해\n'
 '- ③ 전쟁, 외국의 무력행사, 혁명, 내란, 사변, 폭동, 소'),
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
