from langchain_core.documents import Document

chunk = Document(
    page_content=('등에서 치료가 곤란<br>하여 의료기관에 입실하여 의사의 관리하에 치료에 전념하는 것을 말합니다.<br>\uf000 제1항의 '
 '"의료기관"이라 함은 의료법 제3조(의료기관) 제2항에서 정한 국내의 병</p><br><p id=\'51\' '
 "data-category='paragraph' style='font-size:16px'>원이나 의원 또는 국외의 의료관련법에서 정한 "
 "의료기관을 말합니다.</p><p id='52' data-category='paragraph' "
 "style='font-size:16px'>제5조(특별약관의 소멸)<br>피보험자가"),
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
