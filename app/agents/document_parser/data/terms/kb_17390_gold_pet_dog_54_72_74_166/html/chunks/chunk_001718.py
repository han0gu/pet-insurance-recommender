from langchain_core.documents import Document

chunk = Document(
    page_content=('. 시행) 중 다음에 적은 상병을 말하며 이후<br>한국표준질병․사인분류가 개정되는 경우는 개정된 기준에 따라 이 약관에서 '
 "보장</h1><br><table id='50' style='font-size:14px'><thead><tr><td>하는 골절 해당 여부를 "
 '판단합니다.</td><td></td></tr></thead><tbody><tr><td>대상이 되는 '
 '항목</td><td>분류번호</td></tr><tr><td>두개골 및 안면골의 '
 '골절</td><td>S02</td></tr><tr><td>머리의'),
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
