from langchain_core.documents import Document

chunk = Document(
    page_content=('. 장해의 분류</td><td>공 통</td></tr></thead><tbody><tr><td>장해의 분류 '
 '지급률</td><td>사항</td></tr><tr><td>1) 신경계에 장해가 남아 일상생활 기본동작에 제한을 남긴 '
 '때</td><td>10~100</td></tr><tr><td>2) 정신행동에 극심한 장해를 '
 '남긴때</td><td>100</td></tr><tr><td>3) 정신행동에 심한 장해를 남긴 때</td><td>75 '
 '보</td></tr><tr><td>4) 정신행동에 뚜렷한 장해를 남긴 때</td><td>50'),
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
