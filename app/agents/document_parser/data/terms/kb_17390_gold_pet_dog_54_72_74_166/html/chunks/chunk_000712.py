from langchain_core.documents import Document

chunk = Document(
    page_content=('. 제3자는 의료법 제3조(의료기관)에 규정한 종합병원 소속 전문의 중<br>에 정하며, 보험금 지급사유 판정에 드는 의료비용은 회사가 '
 '전액 부담합니다.<br>\uf000 피보험자가 보험기간 중 사망하고, 그 후에 제3조(특정정신질환의 정의 및 진단<br>확정)에서 정한 '
 '"특정정신질환"을 직접적인 원인으로 사망한 사실이 확인된 경우<br>에는 그 사망일을 진단 확정일로 보고 제1조(보험금의 지급사유)에 '
 '해당하는 경<br>우에 한하여 해당 보험금을 지급합니다'),
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
