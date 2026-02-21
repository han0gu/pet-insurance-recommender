from langchain_core.documents import Document

chunk = Document(
    page_content=('- 해\n'
 '- 를 수 있습니다. 제3자는 의료법 제3조(의료기관)에 규정한 종합병원 소속 전문 및\n'
 '- 의 중에 정하며, 보험금 지급사유 판정에 드는 의료비용은 회사가 전액 부담합니 질\n'
 '- 다. 병\n'
 '제3조(특별약관의 소멸)피보험자가 사망하였을반\n'
 '해약환급금 산출방법서"에서 정하는 바에 따라 피보험자의 사망 당시 이 특별약관의 려\n'
 '계약자적립액 및 미경과보험료를 계약자에게 지급합니다. 동'),
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
