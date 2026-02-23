from langchain_core.documents import Document

chunk = Document(
    page_content=('- ② 제1항의 규정에 따라 해지하지 않은 계약은 파산선고 후 3개월이 지난 때에는 그 효\n'
 '- 력을 잃습니다.\n'
 '- ③ 제1항의 규정에 따라 계약이 해지되거나 제2항의 규정에 따라 계약이 효력을 잃는 경\n'
 '- 우에 회사는 제33조(해약환급금) 제1항에 의한 해약환급금을 계약자에게 지급합니다.\n'
 '# 제33조 (해약환급금)- ① 이 약관에 따른 해약환급금은 "보험료 및 해약환급금 산출방법서" 에 따라 계산합니\n'
 '- 다.\n'
 '- ② 해약환급금의 지급사유가 발생한 경우 계약자는 회사에 해약환급금을 청구하여야 하'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
