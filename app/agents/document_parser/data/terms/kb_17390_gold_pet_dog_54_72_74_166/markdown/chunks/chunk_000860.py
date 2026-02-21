from langchain_core.documents import Document

chunk = Document(
    page_content=('시 흡인이 발생하고 연식 외에는 섭취가 불가능한 상태\n'
 '4) ‘씹어먹는 기능에 약간의 장해를 남긴 때’라 함은 아래의 경우 중 하나- \n'
 '이상에 해당되는 때를 말한다.\n'
 '가) 약간의 개구(입벌리기)운동 제한 또는 약간의 저작(씹기)운동 제한\n'
 '공\n'
 '으로 부드러운 고형식(밥, 빵 등)만 섭취 가능한 경우\n'
 '통\n'
 '나) 위‧아래턱(상․하악)의 가운데 앞니(중절치)간 최대 개구(입벌리기)\n'
 '운동이 2cm이하로 제한되는 경우 사항\n'
 '다) 위‧아래턱(상․하악)의 부정교합(전방, 측방)이 1cm이상인 경우'),
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
