from langchain_core.documents import Document

chunk = Document(
    page_content=('계약대출금으로 보험료가 자동으로 납입되어 계약은 유효하게 지속됩니다. 다만, 계약\n'
 '자가 서면 이외에 인터넷 또는 전화(음성녹음) 등으로 자동대출납입을 신청할 경우 회\n'
 '사는 자동대출납입 신청내역을 서면, 전화(음성녹음) 또는 전자문서(SMS포함) 등으로\n'
 '계약자에게 알려 드립니다.<용어풀이># [자동대출납입]보험료를 제때에 납입하기 곤란한 경우에 계약자가 자동대출납입을 신청하면 해당 보험 '
 '상품의 해\n'
 '약환급금 범위 내에서 납입할 보험료를 자동적으로 대출하여 이를 보험료 납입에 충당하는 서비스'),
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
