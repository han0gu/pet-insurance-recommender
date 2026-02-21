from langchain_core.documents import Document

chunk = Document(
    page_content=('- 제2천추 이하의 천골, 미골, 좌골 포함), 빗장뼈(쇄\n'
 '- 골), 가슴뼈(흉골), 갈비뼈(늑골)를 말하며 이를 모두\n'
 '- 동일한 부위로 본다.\n'
 '- 2) “골반뼈의 뚜렷한 기형”이라 함은 아래의 경우 중\n'
 '213# 하나에 해당하는 때를 말한다.- 가) 천장관절 또는 치골문합부가 분리된 상태로 치유\n'
 '- 되었거나 좌골이 2.5cm 이상 분리된 부정유합 상태\n'
 '- 나) 육안으로 변형(결손을 포함)을 명백하게 알 수\n'
 '- 있을 정도로 방사선 검사로 측정한 각(角) 변형\n'
 '- 이 20° 이상인 경우'),
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
