from langchain_core.documents import Document

chunk = Document(
    page_content=('- 하고, 이로 인하여 발생하는 추가 또는 환급되는 보험료는 보험료 및 해약환급금 산출방법서에서\n'
 '- 정한 바에 따라 일단위로 계산하여 받거나 돌려 드립니다.\n'
 '# 제6조(보험증권의 발급)- ① 회사는 계약자에게 보험증권을 드려야 하고, 그 약관의 주요한 내용을 알려드립니다.\n'
 '- ② 계약자의 요청이 있을 경우, 개별 피보험자에게는 가입증명서를 발급하여 드립니다.\n'
 '# 제7조(적용상의 특칙)계약자가 아닌 단체의 소속원이 보험료 전부 또는 일부를 부담하는 경우에는 그 소속원이 계약자로서'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
