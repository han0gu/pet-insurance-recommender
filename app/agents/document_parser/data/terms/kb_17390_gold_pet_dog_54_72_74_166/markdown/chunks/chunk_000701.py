from langchain_core.documents import Document

chunk = Document(
    page_content=('- 1. 계약자, 피보험자 또는 이들의 대리인이 제16조(계약 전 알릴 의무)에도 불구\n'
 '- 하고 고의 또는 중대한 과실로 중요한 사항에 대하여 사실과 다르게 알린 때\n'
 '- 2. 뚜렷한 위험의 변경 또는 증가와 관련된 제17조(계약 후 알릴 의무)에서 정한\n'
 '- 계약 후 알릴 의무를 계약자 또는 피보험자의 고의 또는 중대한 과실로 이행\n'
 '하지 않았을 때\n'
 '\uf000 제1항 제1호의 경우에도 불구하고 다음 중 하나에 해당하는 경우에는 회사는 계- \uf000 제1항에 의한 계약의 해지가 '
 '손해발생 전에 이루어진 경우에는 보통약관 제1절'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
