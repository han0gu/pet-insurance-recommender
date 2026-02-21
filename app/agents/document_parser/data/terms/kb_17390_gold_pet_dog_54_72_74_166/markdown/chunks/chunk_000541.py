from langchain_core.documents import Document

chunk = Document(
    page_content=('- 자가 재가입을 원하지 않는 경우에는 해당 시점으로부터 계약은 해지됩니다(단,\n'
 '- 최초연장된 날로부터 90일 이전에는 계약을 취소 또는 해지할 수 있습니다.)\n'
 '- \uf000 제7항 내지 제9항에 따라 계약이 해지된 경우 회사는 보통약관 제1절 일반조항\n'
 '- \uf000\n'
 '- 108 -제34조제1항에 따른 해약환급금을 계약자에게 지급합니다.제23조(준용규정)\n'
 '반려동물(강아지) 일반조항에서 정하지 않은 사항은 보통약관 제1절 일반조항을 따'),
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
