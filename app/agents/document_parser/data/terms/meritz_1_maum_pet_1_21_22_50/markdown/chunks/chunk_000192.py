from langchain_core.documents import Document

chunk = Document(
    page_content=('증가 감소 또는 교체) 제2항에도 불구하고 새로이 증가되는 보험의 목적의 보험기간을 계\n'
 '약의 남은 보험기간과 다르게 정하는 경우에 적용합니다.# 제2조(보험기간)이 추가특별약관에 따라 계약기간 중에 새로이 증가된 보험의 '
 '목적의 보험기간은 계약자가\n'
 '요청하는 기간으로 합니다.# 제3조(보험료의 납입)- ① 계약자는 새로이 증가된 보험의 목적에 대하여 일단위로 계산된 추가보험료를 '
 '납입하여\n'
 '- 야 합니다.\n'
 '- ② 새로이 증가된 보험의 목적의 보험기간이 시작된 후라도 다른 약정이 없으면 추가 보험'),
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
