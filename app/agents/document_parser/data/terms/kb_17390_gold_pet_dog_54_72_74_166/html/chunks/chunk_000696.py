from langchain_core.documents import Document

chunk = Document(
    page_content=('분류표)에서 정한 질환을 말합니다.<br>\uf000 제1항의 "환경성질환"의 진단확정은 의료법 제3조에서 정한 병원 또는 국외의 '
 '의<br>료관련법에서 정한 의료기관의 의사자격을 가진 자에 의한 진단서에 의합니다'),
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
