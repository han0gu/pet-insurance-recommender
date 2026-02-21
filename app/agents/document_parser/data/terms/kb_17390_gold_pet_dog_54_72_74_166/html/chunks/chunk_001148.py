from langchain_core.documents import Document

chunk = Document(
    page_content=("id='165' data-category='paragraph' style='font-size:18px'>- 120 -</p><p "
 "id='166' data-category='index' style='font-size:14px'>용 어 정 의 부 가 설 명<br>∙ "
 '핵연료물질 : 사용된 연료를 포함합니다.<br>배상책임에 있어 회사와 계약자간에 약정한 금액으로 피보 특<br>∙ 핵연료물질에 의하여 '
 '오염된 물질 : 원자핵 분열 생성물을 포함합니다'),
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
