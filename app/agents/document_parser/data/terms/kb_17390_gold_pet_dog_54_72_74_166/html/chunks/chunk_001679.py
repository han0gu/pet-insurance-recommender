from langchain_core.documents import Document

chunk = Document(
    page_content=("금쪽같은 펫보험(강아지)(무배당)(26.01) 155</p><br><p id='22' data-category='paragraph' "
 "style='font-size:18px'>- 155 -</p><table id='23' "
 "style='font-size:14px'><thead></thead><tbody><tr><td></td><td>유형 제한 정도 "
 '상태,</td><td>지급률 20%</td></tr><tr><td rowspan="4">배변·</td><td>1) 배설을 돕기 위해 '
 '설치한 의료장치나 외과적 시술물을 사용함에 있어 타인의'),
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
