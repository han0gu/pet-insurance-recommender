from langchain_core.documents import Document

chunk = Document(
    page_content=('- 다) 근전도 검사상 불완전한 손상(incomplete injury)소견이 있으면서 도\n'
 '- 수근력검사(MMT)에서 근력이 3등급(fair)인 경우\n'
 '- 11) 동요장해 평가 시에는 정상측과 환측을 비교하여 증가된 수치로 평가한다.\n'
 '12) ‘가관절주 \ue045 이 남아 뚜렷한 장해를 남긴 때’라 함은 대퇴골에 가관절이\n'
 '남은 경우 또는 경골과 종아리뼈의 2개 뼈 모두에 가관절이 남은 경우를\n'
 '공\n'
 '말한다.\n'
 '통\n'
 '주) 가관절이란, 충분한 경과 및 골이식술 등 골유합을 얻는데 필요한\n'
 '수술적 치료를 시행하였음에도 불구하고 골절부의 유합이 이루어지 사항'),
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
