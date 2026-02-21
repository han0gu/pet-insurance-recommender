from langchain_core.documents import Document

chunk = Document(
    page_content=('. 시행) 중 다음에 적은 질병을 말<br>하며 이후 한국표준질병․사인분류가 개정되는 경우는 개정된 기준에 따라 이 '
 "약관</p><br><table id='108' "
 "style='font-size:16px'><thead><tr><td>에서</td><td>보장하는 환경성질환 해당 여부를 "
 '판단합니다.</td><td></td></tr></thead><tbody><tr><td>구분</td><td>대상이 되는 '
 '질병</td><td>분류번호</td></tr><tr><td>아토피 알레르기성</td><td>아토피성'),
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
