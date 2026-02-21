from langchain_core.documents import Document

chunk = Document(
    page_content=('- 해지되었으나 해약환급금을 받지 않은 경우(보험계약대출 등에 따라 해약환급금이 차\n'
 '- 감되었으나 받지 않은 경우 또는 해약환급금이 없는 경우를 포함합니다) 계약자는 해\n'
 '- 지된 날부터 3년 이내에 회사가 정한 절차에 따라 계약의 부활(효력회복)을 청약할 수\n'
 '- 있습니다. 회사가 부활(효력회복)을 승낙한 때에 계약자는 부활(효력회복)을 청약한\n'
 '- 날까지의 연체된 보험료에 평균공시이율+1% 범위 내에서 각 상품별로 회사가 정하는\n'
 '- 이율로 계산한 금액을 더하여 납입하여야 합니다. 다만, 금리연동형보험은 각 상품별'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
