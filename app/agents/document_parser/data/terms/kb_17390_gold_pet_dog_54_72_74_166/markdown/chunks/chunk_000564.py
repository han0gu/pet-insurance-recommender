from langchain_core.documents import Document

chunk = Document(
    page_content=('- 약, 의약부외품 등)\n'
 '- 11. 한의학(단, 침구는 제외합니다.), 인도의학, 허브요법, 아로마테라피 등의 대\n'
 '- 체의료 및 재활치료\n'
 '- 12. 목욕비용(약용 및 처방샴푸 값 포함) 및 귀세정제(이어클리너), 예방 가능한\n'
 '- 기생충(벼룩, 젝켄, 모공충 등)의 제거비용 및 기생충으로 발생한 질병의 치료\n'
 '- 비\n'
 '- 13. 펫호텔 비용 또는 위탁료, 산책료, 카운슬링 비용, 상담료, 지도료 및 이와 동\n'
 '- 종의 비용\n'
 '- 14. 왕진료, 가입동물의 이송비, 동물병원에 가지 않고 약제만 배달되는 배달료 및\n'
 '- 이와 동종의 비용'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
