from langchain_core.documents import Document

chunk = Document(
    page_content=('발생한 후 3년간 보험금을 청구하지 않는 경우 보험금을 지급받지 못할 수 있습니다.\n'
 '(이하 같습니다.)# 제37조(약관의 해석)① 회사는 신의성실의 원칙에 따라 공정하게 약관을 해석하여야 하며 계약자에 따라 다르\n'
 '게 해석하지 않습니다.# 【신의성실의 원칙】권리의 행사와 의무의 이행은 신의와 성실을 가지고 행동하여 상대방의 신뢰와 기대\n'
 '를 배반하여서는 안 된다는 원칙(「민법」제2조 제1항)- ② 회사는 약관의 뜻이 명백하지 않은 경우에는 계약자에게 유리하게 해석합니다.'),
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
