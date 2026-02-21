from langchain_core.documents import Document

chunk = Document(
    page_content=('경된 이율을 적용합니다.- 약\n'
 '- 1. "보장성-1701 공시이율"은 매월 마지막날 회사가 정한 이율로 하며, 다음달 1일 관\n'
 '- 부터 마지막날까지 1개월간 확정 적용합니다.\n'
 '- 2. 회사는 외부지표금리와 운용자산이익률을 가중평균하여 산출된 공시기준이율에\n'
 '- 향후 예상수익 등을 고려한 조정률을 적용하여 "보장성-1701 공시이율"을 결정\n'
 '- 합니다.\n'
 '- 3. "보장성-1701 공시이율"의 최저보증이율은 연단위 복리 0.2%를 적용합니다. 별\n'
 '- 표'),
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
