from langchain_core.documents import Document

chunk = Document(
    page_content=('- 벽을 오르내리거나 특수한 기술, 경험, 사전훈련을 필\n'
 '- 요로 하는 등반을 말합니다), 글라이더 조종, 스카이\n'
 '- 다이빙, 스쿠버다이빙, 행글라이딩, 수상보트, 패러글\n'
 '- 라이딩\n'
 '- ② 모터보트, 자동차 또는 오토바이에 의한 경기, 시범,\n'
 '- 흥행(이를 위한 연습을 포함합니다) 또는 시운전(다\n'
 '- 만, 공용도로상에서 시운전을 하는 동안 보험금 지급\n'
 '- 사유가 발생한 경우에는 보장합니다)\n'
 '- ③ 선박에 탑승하는 것을 직무로 하는 사람이 직무상 선\n'
 '- 박에 탑승하고 있는 동안'),
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
