from langchain_core.documents import Document

chunk = Document(
    page_content=('이를 알리지 않았을 때에는 그 타인은 이 계약이 체결된 사\n'
 '실을 알지 못하였다는 사유로 회사에 이의를 제기 할 수 없\n'
 '습니다.\n'
 '\uf000 타인을 위한 계약에서 보험사고가 발생한 경우에 계약자\n'
 '가 그 타인에게 보험사고의 발생으로 생긴 손해를 배상한\n'
 '때에는 계약자는 그 타인의 권리를 해하지 않는 범위 안에\n'
 '서 회사에 보험금의 지급을 청구할 수 있습니다.# 제17조(준용규정)이 특별약관에서 정하지 않은 사항은「반려동물 비용손해\n'
 '관련 특별약관 일반조항」을 따릅니다.「반려동물 비용손해\n'
 '관련 특별약관 일반조항」에서 정하지 않은 사항은 보통약'),
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
