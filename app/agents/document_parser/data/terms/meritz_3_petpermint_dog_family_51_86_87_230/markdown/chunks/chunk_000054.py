from langchain_core.documents import Document

chunk = Document(
    page_content=('발생을 막을 수 있었음에도 그 주의조차 태만히 한 높은\n'
 '강도의 주의의무위반# 제17조(알릴 의무 위반의 효과)\uf000 회사는 아래와 같은 사실이 있을 경우에는 손해의 발생\n'
 '여부에 관계없이 이 계약을 해지할 수 있습니다.- ① 계약자 또는 피보험자가 고의 또는 중대한 과실로 제\n'
 '- 15조(계약 전 알릴 의무)를 위반하고 그 의무가 중요\n'
 '- 한 사항에 해당하는 경우\n'
 '- ② 뚜렷한 위험의 증가와 관련된 제16조(상해보험계약 후\n'
 '- 알릴 의무) 제1항에서 정한 계약 후 알릴 의무를 계약\n'
 '- 자 또는 피보험자의 고의 또는 중대한 과실로 이행하'),
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
