from langchain_core.documents import Document

chunk = Document(
    page_content=('별약관이 해지되었으나 해약환급금을 받지 않은 경우(보험계약대출 등에 따라 해약환\n'
 '급금이 차감되었으나 받지 않은 경우 또는 해약환급금이 없는 경우를 포함합니다) 계\n'
 '약자는 해지된 날부터 3년 이내에 회사가 정한 절차에 따라 특별약관의 부활(효력회\n'
 '복)을 청약할 수 있습니다. 회사가 부활(효력회복)을 승낙한 때에는 계약자는 부활(효\n'
 '력회복)을 청약한 날까지의 연체된 보험료에 평균공시이율 + 1% 범위 내에서 각 상품\n'
 '별로 회사가 정하는 이율로 계산한 금액을 더하여 납입하여야 합니다. 다만, 금리연동'),
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
