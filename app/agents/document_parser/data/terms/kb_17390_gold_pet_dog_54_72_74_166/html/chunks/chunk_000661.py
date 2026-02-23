from langchain_core.documents import Document

chunk = Document(
    page_content=("등을 차감하고 그 차액을</h1><br><p id='212' data-category='list' "
 "style='font-size:14px'>제3조(천식지속상태의 정의 및 진단 확정)<br>\uf000 이 특별약관에 있어서 "
 '"천식지속상태"라 함은 제9차 한국표준질병․사인분류에 있<br>어서 【별표13】(천식지속상태 분류표)에서 정한 질병을 '
 '말합니다.<br>\uf000 "천식지속상태"의 진단확정은 의료법 제3조(의료기관)에서 정한 국내의 병원이나<br>의원 또는 국외의 '
 '의료관련법에서 정한 의료기관의 의사(치과의사 제외) 면허를<br>가진 자에 의하여'),
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
