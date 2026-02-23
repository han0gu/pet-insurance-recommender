from langchain_core.documents import Document

chunk = Document(
    page_content=('계약자로서의 권리를 행사할 수 있습니다.제8조(준용규정)이 특별약관에 정하지 않은 사항은 보통약관을 따릅니다.- 38 -단체계약 '
 '보험료정산 추가특별약관제1조(보험료의 정산)① 회사는 단체계약 특별약관(이하“특별약관”이라 합니다) 제4조(보험의 목적의 증가 감소\n'
 '또는 교체) 제2항 및 보통약관 제16조(계약 후 알릴 의무) 제2항에도 불구하고 이 추가\n'
 '특별약관에 따라 보험료를 정산합니다.\n'
 '② 회사는 특별약관 제4조(보험의 목적의 증가 감소 또는 교체) 제3항에도 불구하고 보험'),
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
