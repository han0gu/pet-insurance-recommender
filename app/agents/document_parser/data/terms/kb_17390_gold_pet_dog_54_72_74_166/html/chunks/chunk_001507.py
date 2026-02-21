from langchain_core.documents import Document

chunk = Document(
    page_content=('이상 및 장해상태를 평가<br>하기 위해 아래의 검사들을 기초로 한다.<br>가) 뇌영상검사(CT, MRI)<br>나) 온도안진검사, '
 "전기안진검사(또는 비디오안진검사) 등</p><p id='197' data-category='paragraph' "
 "style='font-size:14px'>142 KB 금쪽같은 펫보험(강아지)(무배당)(26.01)</p><br><p id='198' "
 "data-category='list'></p><p id='199' data-category='paragraph'"),
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
