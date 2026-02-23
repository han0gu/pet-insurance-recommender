from langchain_core.documents import Document

chunk = Document(
    page_content=('- 해당하는 경우를 말한다.\n'
 '- 가) 완전 강직(관절굳음)\n'
 '- 나) 근전도 검사상 완전손상(complete injury) 소견이 있으면서 도수근\n'
 '- 력검사(MMT)에서 근력이 ‘0등급(zero)’인 경우\n'
 '- 8) ‘관절 하나의 기능에 심한 장해를 남긴 때’라 함은 아래의 경우 중 하\n'
 '- 나에 해당하는 경우를 말한다.\n'
 '- 가) 해당 관절의 운동범위 합계가 정상 운동범위의 1/4 이하로 제한된 경우\n'
 '- 나) 인공관절이나 인공골두를 삽입한 경우\n'
 '- 다) 근전도 검사상 완전손상(complete injury)소견이 있으면서 도수근'),
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
