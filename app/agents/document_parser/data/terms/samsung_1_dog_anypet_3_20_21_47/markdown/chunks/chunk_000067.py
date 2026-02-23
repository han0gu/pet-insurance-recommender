from langchain_core.documents import Document

chunk = Document(
    page_content=('를 이행하는 것을 말합니다.\n'
 '【담보권 실행】 담보권을 설정한 채권자가 채무를 이행하지 않는 채무자에 대하여 해당 담보권을 실행하\n'
 '는 것을 말합니다. 법원은 채권자의 신청에 따른 강제집행 및 담보권실행으로 채무자의 환급금을 압류할\n'
 '수 있으며, 법원의 추심명령 또는 전부명령에 따라 회사는 채권자에게 환급금을 지급하게 됩니다.\n'
 '【국세 및 지방세 체납처분 절차】 국세 또는 지방세를 체납할 경우 국세 기본법 및 지방세법에 의하여 체\n'
 '납된 세금에 대하여 가산금 징수, 독촉장 발부 및 재산 압류 등의 집행을 하는 것을 말합니다. 국세 및 지'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
