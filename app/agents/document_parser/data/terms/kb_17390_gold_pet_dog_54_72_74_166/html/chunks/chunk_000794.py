from langchain_core.documents import Document

chunk = Document(
    page_content=("금쪽같은 펫보험(강아지)(무배당)(26.01)</p><br><table id='160' "
 "style='font-size:14px'><thead></thead><tbody><tr><td></td></tr><tr><td>관 련 법 "
 '규 수의사법 제2조(정의) 이 법에서 사용하는 용어의 뜻은 다음과 같다'),
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
