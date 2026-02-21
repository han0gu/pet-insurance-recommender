from langchain_core.documents import Document

chunk = Document(
    page_content=('. 제3자는 의료법 제3조(의료기관)에 규정한 종합병원 소속 전문<br>의 중에 정하며, 보험금 지급사유 판정에 드는 의료비용은 회사가 '
 '전액 부담합니<br>다.<br>\uf000 같은 상해로 두 가지 이상의 후유장해가 생긴 경우에는 후유장해 지급률을 합산<br>하여 '
 '지급합니다. 다만, 장해분류표의 각 신체부위별 판정기준에 별도로 정한 경<br>우에는 그 기준에 따릅니다.<br>\uf000 다른 상해로 '
 '인하여 후유장해가 2회 이상 발생하였을 경우에는 그 때마다 이에 해<br>당하는 후유장해지급률을 결정합니다'),
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
