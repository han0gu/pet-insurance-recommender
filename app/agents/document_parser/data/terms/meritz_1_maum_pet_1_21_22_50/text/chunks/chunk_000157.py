from langchain_core.documents import Document

chunk = Document(
    page_content=('반환일까지의 기간에 대하여 회사는 보험개발원이 공시하는 보험계약대출이율을 연단위 복\n'
 '리로 계산한 금액을 더하여 돌려 드립니다.제18조(타인을 위한 계약)① 계약자는 타인을 위한 계약을 체결하는 경우에 그 타인의 위임이 '
 '없는 때에는 반드시\n'
 '이를 회사에 알려야 하며, 이를 알리지 않았을 때에는 그 타인은 이 계약이 체결된 사\n'
 '실을 알지 못하였다는 사유로 회사에 이의를 제기할 수 없습니다.\n'
 '② 타인을 위한 계약에서 보험사고가 발생한 경우에 계약자가 그 타인에게 보험사고의 발'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
