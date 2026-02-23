from langchain_core.documents import Document

chunk = Document(
    page_content=('자동대출납입을 신청할 수 있으며, 이 경우 제37조(보험계약대출) 제1항에 따른 보험\n'
 '계약대출금으로 보험료가 자동으로 납입되어 계약은 유효하게 지속됩니다. 다만, 계약\n'
 '자가 서면 이외에 인터넷 또는 전화(음성녹음) 등으로 자동대출납입을 신청할 경우 회\n'
 '사는 자동대출납입 신청내역을 서면, 전화(음성녹음) 또는 전자문서(SMS포함) 등으로\n'
 '계약자에게 알려 드립니다.<용어풀이># [자동대출납입]보험료를 제때에 납입하기 곤란한 경우에 계약자가 자동대출납입을 신청하면 해당 보험 '
 '상품의 해'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
