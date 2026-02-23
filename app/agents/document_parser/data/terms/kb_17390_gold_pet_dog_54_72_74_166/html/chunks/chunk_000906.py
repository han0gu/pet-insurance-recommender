from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:16px'>계약자는 특별약관이 소멸하기</p><br><p id='81' "
 "data-category='paragraph' style='font-size:14px'>회사는 보통약관 제1절 일반조항 "
 '제34조(해약환급금) 제1항에 따른 해약환급금을 계 성<br>약자에게 지급합니다'),
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
