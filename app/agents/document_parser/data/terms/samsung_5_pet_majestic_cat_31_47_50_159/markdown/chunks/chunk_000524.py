from langchain_core.documents import Document

chunk = Document(
    page_content=('않는 채무자에 대하여 해당 담보권을 실행하는 것을 말합니다.\n'
 '법원은 채권자의 신청에 따른 강제집행 및 담보권실행으로 채무자의 해약환급금을 압류할 수 있으\n'
 '며, 법원의 추심명령 또는 전부명령에 따라 회사는 채권자에게 해약환급금을 지급하게 됩니다.- 추심명령 : 채무자가 제3채무자에 대하여 '
 '가지고 있는 금전채권을 대위의 절차 없이 채무자\n'
 '- 에 갈음하여 직접 추심(받아냄)할 수 있는 권리를 부여하는 집행법원의 결정\n'
 '- 전부명령 : 채무자가 제3채무자에 대한 채권을 채권자에게 이전시키고 그 대신 채무자에 대'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
