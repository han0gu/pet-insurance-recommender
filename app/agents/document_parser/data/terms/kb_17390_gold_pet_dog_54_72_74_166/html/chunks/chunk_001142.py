from langchain_core.documents import Document

chunk = Document(
    page_content=("기관)을 말합니다.</td></tr></tbody></table><p id='161' data-category='paragraph' "
 "style='font-size:14px'>120 KB 금쪽같은 펫보험(강아지)(무배당)(26.01)</p><br><table "
 "id='162' style='font-size:14px'><thead><tr><td>용 "
 '어</td><td>정</td></tr></thead><tbody><tr><td>반려동물</td><td>의 보험증권에 기재된 반려동물을 '
 '말하며, 이 계약에서 가입 가능한 반려동물은 대한민국'),
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
