from langchain_core.documents import Document

chunk = Document(
    page_content=(". 단, 관절기능장해</header><br><p id='91' data-category='list' "
 "style='font-size:14px'>가 신경손상으로 인한 경우에는 운동범위 측정이 아닌 근력 및 근전<br>도 검사를 기준으로 "
 '평가한다.<br>7) ‘관절 하나의 기능을 완전히 잃었을 때’라 함은 아래의 경우 중 하나에<br>해당하는 경우를 말한다.<br>가) '
 '완전 강직(관절굳음)<br>나) 근전도 검사상 완전손상(complete injury) 소견이 있으면서 도수근<br>력검사(MMT)에서 '
 '근력이 ‘0등급(zero)’인'),
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
