from langchain_core.documents import Document

chunk = Document(
    page_content=(". 시행) 중 다음에 적은 질병을</p><br><table id='1' "
 "style='font-size:14px'><thead></thead><tbody><tr><td>말하며 이후 한국표준질병․사인분류가 "
 '개정되는 경우는 이 약</td><td>개정된 기준에 따라</td></tr><tr><td><table><thead><tr><td>관에서 '
 '보장하는 특정정신질환 해당 여부를</td><td>판단합니다.</td></tr></thead><tbody><tr><td>대상이 되는'),
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
