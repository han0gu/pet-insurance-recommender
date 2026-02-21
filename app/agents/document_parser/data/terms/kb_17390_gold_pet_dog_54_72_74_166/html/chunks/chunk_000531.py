from langchain_core.documents import Document

chunk = Document(
    page_content=('"골절진단(치아파절제외)"이라 함은 【별표4】(골절분류<br>표Ⅱ(치아파절제외))에 정한 상병을 말합니다.<br>\uf000 '
 '"골절진단(치아파절제외)"의 진단은 의료법 제3조(의료기관)에서 정한 국내의 병 상<br>원이나 의원 또는 국외의 의료관련법에서 정한 '
 '의료기관의 의사 면허를 가진 자 해<br>에 의하여 내려져야 합니다'),
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
