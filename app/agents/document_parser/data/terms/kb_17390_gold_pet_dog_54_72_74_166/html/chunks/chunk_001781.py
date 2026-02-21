from langchain_core.documents import Document

chunk = Document(
    page_content=('. 시행) 중 다음에 적은 질병을 말하며 이후 한국<br>표준질병․사인분류가 개정되는 경우는 개정된 기준에 따라 이 약관에서 '
 "보장하는</p><br><table id='102' style='font-size:14px'><thead><tr><td>호흡기관련질병 해당 "
 '여부를 판단합니다.</td><td></td></tr></thead><tbody><tr><td>대상이 되는 '
 '항목</td><td>분류번호</td></tr><tr><td>급성상기도감염</td><td>J00~J06</td></tr><tr><td>상기도의 '
 '상세불명 질환 급성인지'),
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
