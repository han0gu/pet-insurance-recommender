from langchain_core.documents import Document

chunk = Document(
    page_content=(". 전쟁, 외국의 무력행사, 혁명, 내란, 사변, 폭동</p><br><table id='43' "
 "style='font-size:14px'><thead></thead><tbody><tr><td>용 어 풀</td><td>이 습관성 유산, "
 '불임 및 인공수정</td></tr><tr><td colspan="2">한국표준질병․사인분류상의 N96~N98에 해당하는 질병을 '
 '말합니다.</td></tr><tr><td>용 어 풀</td><td>이 심신상실</td></tr><tr><td>정신병, 정신박약, 심한 '
 '등의 사물 는 의사 결정 능력이 없는'),
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
