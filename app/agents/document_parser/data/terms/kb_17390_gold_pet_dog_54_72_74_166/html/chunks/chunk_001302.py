from langchain_core.documents import Document

chunk = Document(
    page_content=("미치지 않습니다.<br>용 어 풀 이 공제계약</p><br><table id='142' "
 "style='font-size:14px'><thead></thead><tbody><tr><td>유사보험으로서 공제</td><td>사업을 "
 '실시하는 경영주체와 공제 계약자 사이에 체결되</td></tr><tr><td colspan="2">는 계약을 말합니다'),
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
