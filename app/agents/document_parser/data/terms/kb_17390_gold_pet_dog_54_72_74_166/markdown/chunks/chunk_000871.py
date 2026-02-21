from langchain_core.documents import Document

chunk = Document(
    page_content=('손바닥 크기 이상의 추상(추한 모습)- \n'
 '라.# 약간의- \n'
 '추상(추한 모습)1) 얼굴\n'
 '가) 손바닥 크기 1/4 이상의 추상(추한 모습)\n'
 '나) 길이 5cm 이상의 추상반흔(추한 모습의 흉터)\n'
 '다) 지름 2cm 이상의 조직함몰\n'
 '라) 코의 1/4 이상 결손\n'
 '2) 머리\n'
 '가) 손바닥 크기 1/2 이상의 반흔(흉터) 및 모발결손\n'
 '나) 머리뼈의 손바닥 크기 1/2 이상의 손상 및 결손\n'
 '3) 목\n'
 '손바닥 크기 1/2 이상의 추상(추한 모습)# 마. 손바닥 크기- \n'
 '‘손바닥 크기’라 함은 해당 환자의 손가락을 제외한 손바닥의 크기를 말하'),
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
