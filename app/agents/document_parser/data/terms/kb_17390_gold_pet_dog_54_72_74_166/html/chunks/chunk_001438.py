from langchain_core.documents import Document

chunk = Document(
    page_content=("정하지 않은 사항에 대하여는 전환대상계약 약관, 소득세법 등</p><p id='109' data-category='paragraph' "
 "style='font-size:14px'>138 KB 금쪽같은 펫보험(강아지)(무배당)(26.01)</p><br><h1 id='110' "
 "style='font-size:14px'>관련법규에서 정하는 바에 따릅니다.</h1><br><h1 id='111' "
 "style='font-size:14px'>\uf000 소득세법 등 관련법규가 제·개정 또는 폐지되는 경우 변경된 법령을 "
 "따릅니다.</h1><h1 id='112'"),
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
