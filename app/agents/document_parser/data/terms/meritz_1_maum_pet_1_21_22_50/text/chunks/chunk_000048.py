from langchain_core.documents import Document

chunk = Document(
    page_content=('서에서 질문한 사항에 대하여 알고 있는 사실을 반드시 사실대로 알려야(이하 「계약 전\n'
 '알릴 의무」라 하며, 상법상「고지의무」와 같습니다) 합니다.【계약 전 알릴 의무】상법 제651조에서 정하고 있는 의무. 보험계약자나 '
 '피보험자는 청약시에 보험회사가 서\n'
 '면으로 질문한 중요한 사항에 대해 사실대로 알려야 하며, 위반시 보험계약의 해지 또\n'
 '는 보험금 부지급 등 불이익을 당할 수 있습니다.(이하 같습니다.)제16조(계약 후 알릴 의무)① 계약을 맺은 후 보험의 목적에 아래와 '
 '같은 사실이 생긴 경우에는 계약자나 피보험자는'),
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
