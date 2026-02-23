from langchain_core.documents import Document

chunk = Document(
    page_content=('144 KB 금쪽같은 펫보험(강아지)(무배당)(26.01)- \n'
 '- \n'
 '# 다. 뚜렷한# 1)추상(추한 모습)\n'
 '얼굴\n'
 '가) 손바닥 크기 1/2 이상의 추상(추한 모습)\n'
 '나) 길이 10cm 이상의 추상 반흔(추한 모습의 흉터)\n'
 '다) 지름 5cm 이상의 조직함몰- \n'
 '라) 코의 1/2 이상 결손\n'
 '2) 머리\n'
 '가) 손바닥 크기 이상의 반흔(흉터) 및 모발결손\n'
 '나) 머리뼈의 손바닥 크기 이상의 손상 및 결손\n'
 '3) 목- \n'
 '손바닥 크기 이상의 추상(추한 모습)- \n'
 '라.# 약간의- \n'
 '추상(추한 모습)1) 얼굴'),
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
