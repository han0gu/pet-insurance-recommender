from langchain_core.documents import Document

chunk = Document(
    page_content=('구하고 입원확인서를 변조하여 입원일수 30일에 해당하는\n'
 '보험금을 청구한 경우, 회사는 그 사실을 안 날로부터 1\n'
 '개월 이내에 계약을 해지할 수 있습니다. 다만, 이 경우\n'
 '에도 회사는 입원일수 20일에 해당하는 보험금을 지급합\n'
 '니다.\uf000 회사가 제1항에 따라 계약을 해지한 경우 회사는 그 취\n'
 '지를 계약자에게 통지하고 보통약관 제35조(해약환급금) 제\n'
 '1항에 따른 해약환급금을 지급합니다.# 제22조(준용규정)이「반려동물 비용손해 관련 특별약관 일반조항」에서 정하\n'
 '지 않은 사항은 보통약관을 따릅니다. 다만, 보통약관 제3'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
