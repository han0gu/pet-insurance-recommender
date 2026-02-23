from langchain_core.documents import Document

chunk = Document(
    page_content=('에 따른 보험료의 납입최고(독촉)기간이 지나기 전까지 회사가 정한 방법에 따라 보험\n'
 '료의 자동대출납입을 신청할 수 있으며, 이 경우 제36조(보험계약대출) 제1항에 따른\n'
 '보험계약대출금으로 보험료가 자동으로 납입되어 계약은 유효하게 지속됩니다. 다만,\n'
 '계약자가 서면 이외에 인터넷 또는 전화(음성녹음) 등으로 자동대출납입을 신청할 경\n'
 '우 회사는 자동대출납입 신청내역을 서면, 전화(음성녹음) 또는 전자문서(SMS포함)'),
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
