from langchain_core.documents import Document

chunk = Document(
    page_content=('- "지급에 영향을 미치지 않습니다.\n'
 '- \uf000 보험수익자와 회사가 제1조(보험금의 지급사유)의 보험금 지급사유에 대해 합의\n'
 '- 하지 못할 때는 보험수익자와 회사가 함께 제3자를 정하고 그 제3자의 의견에 따\n'
 '- 를 수 있습니다. 제3자는 의료법 제3조(의료기관)에 규정한 종합병원 소속 전문\n'
 '- 의 중에 정하며, 보험금 지급사유 판정에 드는 의료비용은 회사가 전액 부담합니\n'
 '- 다.\n'
 '# 제3조(특별약관의소멸)- \uf000 회사는 제1조(보험금의 지급사유)에서 정한 반려동물양육자금Ⅱ(질병사망)을 지'),
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
