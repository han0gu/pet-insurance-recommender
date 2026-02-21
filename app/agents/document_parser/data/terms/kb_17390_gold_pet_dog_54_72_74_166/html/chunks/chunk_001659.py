from langchain_core.documents import Document

chunk = Document(
    page_content=('. 단, 질병발생 또는<br>상해를 입은 후 의식상실이 1개월 이상 지속된 경우에는 질병발생<br>또는 상해를 입은 후 12개월이 지난 '
 '후에 판정할 수 있다.<br>나) 정신행동장해는 장해판정 직전 1년 이상 충분한 정신건강의학과의<br>전문적 치료를 받은 후 치료에도 '
 '불구하고 장해가 고착되었을 때 판<br>정하여야 하며, 그렇지 않은 경우에는 그로써 고정되거나 중하게 된<br>장해에 대해서는 인정하지 '
 '않는다.<br>다) ‘정신행동에 극심한 장해를 남긴 때’라 함은 장해판정 직전 1년 이<br>상 지속적인 정신건강의학과의 치료를'),
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
