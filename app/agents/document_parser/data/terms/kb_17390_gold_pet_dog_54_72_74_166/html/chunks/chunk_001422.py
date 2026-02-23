from langchain_core.documents import Document

chunk = Document(
    page_content=('. 장애인전용보험으로 전환을 원할 경우 수익자 지정이 필요합니다.<br>\uf000 전환대상계약이 해지(解止) 또는 기타 사유로 효력이 '
 '없게 된 경우 또는 전환대상<br>계약이 제1항에서 정한 조건을 만족하지 않게 된 경우 이 특별약관은 그 때부터 효<br>력이 '
 '없습니다.<br>\uf000 제2조(제출서류) 제1항에 따라 제출된 장애인증명서상 장애예상기간(또는 장애기<br>질<br>간)이 종료된 '
 '경우에는 제3조(장애인전용보험으로의 전환) 제1항에도 불구하고 이<br>병<br>특별약관은 그때부터 효력이 없습니다.<br>\uf000 '
 '이 특별약관의 계약자는'),
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
