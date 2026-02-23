from langchain_core.documents import Document

chunk = Document(
    page_content=(". 반려동물 특정 질병 보장제한부 인수</h1><p id='113' data-category='paragraph' "
 "style='font-size:14px'>제1조(특별약관의 체결 및 효력)<br>\uf000 이 특별약관은 다음 각 호의 어느 하나에 "
 '해당하는 경우 계약자의 청약과 회사의<br>승낙으로 보험계약(보험약관을 말하며, 특별약관이 부가된 경우에는 그 특별약관<br>을 '
 '포함합니다. 이하 같습니다)에 부가하여 이루어집니다'),
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
