from langchain_core.documents import Document

chunk = Document(
    page_content=('정의 및 진단확정)<br>\uf000 이 특별약관에 있어서 "특정정신질환"이라 함은 제9차 한국표준질병․사인분류에</p><br><p '
 "id='9' data-category='paragraph' style='font-size:14px'>있어서 【별표16】(특정정신질환 "
 '분류표)에서 정한 질병을 말합니다.<br>\uf000 "특정정신질환"의 진단확정은 의료법 제3조(의료기관)에서 정한 국내의 '
 '병원이나<br>의원 또는 국외의 의료관련법에서 정한 의료기관의 정신건강의학과 전문의 자격<br>증을 가진 자에 의하여 작성된 문서화된 '
 '진료기록 또는 검사 결과를'),
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
