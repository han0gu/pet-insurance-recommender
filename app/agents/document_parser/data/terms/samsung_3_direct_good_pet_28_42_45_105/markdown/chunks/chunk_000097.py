from langchain_core.documents import Document

chunk = Document(
    page_content=('⑦ 제1항에 따라 계약이 해지된 경우에는 제33조(해약환급금) 제1항에 따른 해약환급금\n'
 '을 계약자에게 지급합니다.# 제28조 (보험료의 납입을 연체하여 해지된 계약의 부활(효력회복))- ① 제27조(보험료의 납입이 연체되는 '
 '경우 납입최고(독촉)와 계약의 해지)에 따라 계약이\n'
 '- 해지되었으나 해약환급금을 받지 않은 경우(보험계약대출 등에 따라 해약환급금이 차\n'
 '- 감되었으나 받지 않은 경우 또는 해약환급금이 없는 경우를 포함합니다) 계약자는 해\n'
 '- 지된 날부터 3년 이내에 회사가 정한 절차에 따라 계약의 부활(효력회복)을 청약할 수'),
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
