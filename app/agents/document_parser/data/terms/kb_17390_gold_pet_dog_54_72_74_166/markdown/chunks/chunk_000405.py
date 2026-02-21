from langchain_core.documents import Document

chunk = Document(
    page_content=('못할 때는 보험수익자와 회사가 함께 제3자를 정하고 그 제3자의 의견에 따를 수 있\n'
 '습니다. 제3자는 의료법 제3조(의료기관)에 규정한 종합병원 소속 전문의 중에 정하# 며, 보험금 지급사유 판정에 드는 의료비용은 회사가 '
 '전액 부담합니다.# 제3조(호흡기관련질병의 정의 및 진단확정)\uf000 이 특별약관에 있어서 "호흡기관련질병"이라 '
 '함은【별표14】(호흡기관련질병 분\n'
 '류표)에서 정한 질병을 말합니다.\n'
 '\uf000 "호흡기관련질병"의 진단확정은 의료법 제3조에서 정한 국내의 병원이나 의원 또'),
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
