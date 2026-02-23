from langchain_core.documents import Document

chunk = Document(
    page_content=('| 7) 씹어먹는 기능 또는 말하는 기능에 약간의 장 해를 남긴 때 | 5 |\n'
 '| 8) 치아에 14개 이상의 결손이 생긴 때 | 20 |\n'
 '| 9) 치아에 7개 이상의 결손이 생긴 때 | 10 |\n'
 '| 10) 치아에 5개 이상의 결손이 생긴 때 | 5 |\n'
 '206# 나. 장해의 평가기준- 1) 씹어먹는 기능의 장해는 윗니(상악치아)와 아랫니(하\n'
 '- 악치아)의 맞물림(교합), 배열상태 및 아래턱의 개구\n'
 '- (입을 벌림)운동, 삼킴(연하)운동 등에 따라 종합적\n'
 '- 으로 판단하여 결정한다.'),
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
