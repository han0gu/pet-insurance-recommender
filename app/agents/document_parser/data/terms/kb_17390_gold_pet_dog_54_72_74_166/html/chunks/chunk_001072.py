from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:14px'>\uf000 보험의 목적이 다수인 경우 제1항은 보험의 목적별로 각각 적용합니다.</p><p "
 "id='48' data-category='paragraph' style='font-size:14px'>제8조(보험료의 납입을 연체하여 "
 '해지된 계약의 부활(효력회복))<br>부활(효력회복)되는 계약의 보장개시는 반려동물(강아지) 일반조항 제17조(보험료<br>의 납입을 '
 '연체하여 해지된 특별약관의 부활(효력회복))를 따릅니다'),
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
