from langchain_core.documents import Document

chunk = Document(
    page_content=('비용은 회사가 전액 부담합니다.\n'
 '\uf000 같은 상해로 두 가지 이상의 후유장해가 생긴 경우에는\n'
 '후유장해 지급률을 합산하여 지급합니다. 다만,【별표2(장\n'
 '해분류표)】의 각 신체부위별 판정기준에 별도로 정한 경우\n'
 '에는 그 기준에 따릅니다.\n'
 '\uf000 다른 상해로 인하여 후유장해가 2회 이상 발생하였을 경54우에는 그 때마다 이에 해당하는 후유장해지급률을 결정합\n'
 '니다. 그러나 그 후유장해가 이미 후유장해보험금을 지급받\n'
 '은 동일한 부위에 가중된 때에는 최종 장해상태에 해당하는\n'
 '후유장해보험금에서 이미 지급받은 후유장해보험금을 차감'),
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
