from langchain_core.documents import Document

chunk = Document(
    page_content=('- 를 받은 날을 말합니다)부터 15일 이내에 제1항의 절차를 이행할 수 있습니다.\n'
 '# 【강제집행】사법상 또는 행정법상의 의무를 이행하지 아니하는 사람에 대하여 국가가 강제 권\n'
 '력으로 그 의무의 이행하는 것을 말합니다.# 【담보권실행】담보권을 설정한 채권자가 채무를 이행하지 아니하는 채무자에 대하여 해당 담보권\n'
 '을 실행하는 것을 말합니다.# 【국세 및 지방세 체납처분 절차】국세 또는 지방세를 체납할 경우 국세 기본법 및 지방세법에 의하여 체납된 '
 '세금에'),
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
