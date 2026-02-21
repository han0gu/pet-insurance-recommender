from langchain_core.documents import Document

chunk = Document(
    page_content=('그 의무를 이행하는 것을 말합니다. 담보권실행이란 담보권을 설정한 채권자가 채무를 이행하지\n'
 '않는 채무자에 대하여 해당 담보권을 실행하는 것을 말합니다.\n'
 '법원은 채권자의 신청에 따른 강제집행 및 담보권실행으로 채무자의 해약환급금을 압류할 수 있으\n'
 '며, 법원의 추심명령 또는 전부명령에 따라 회사는 채권자에게 해약환급금을 지급하게 됩니다.- · 추심명령 : 채무자가 제3채무자에 대하여 '
 '가지고 있는 금전채권을 대위의 절차 없이 채무자\n'
 '- 에 갈음하여 직접 추심(받아냄)할 수 있는 권리를 부여하는 집행법원의 결정'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
