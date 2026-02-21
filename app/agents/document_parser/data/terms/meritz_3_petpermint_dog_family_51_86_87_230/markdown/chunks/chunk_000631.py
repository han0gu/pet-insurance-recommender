from langchain_core.documents import Document

chunk = Document(
    page_content=('- 다) 머리뼈(두개골)와 상위목뼈(상위경추: 제1, 2경\n'
 '- 추) 사이에 CT 검사 상, 두개 대후두공의 기저점\n'
 '- (basion)과 축추 치돌기 상단사이의 거리(BDI :\n'
 '- Basion-Dental Interval)에 뚜렷한 이상전위가\n'
 '- 있는 상태\n'
 '- 라) 상위목뼈(상위경추: 제1, 2경추) CT 검사상, 환\n'
 '- 추 전방 궁(arch)의 후방과 치상돌기의 전면과의\n'
 '- 거리(ADI: Atlanto-Dental Interval)에 뚜렷한\n'
 '- 이상전위가 있는 상태\n'
 '- 8) 약간의 운동장해'),
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
