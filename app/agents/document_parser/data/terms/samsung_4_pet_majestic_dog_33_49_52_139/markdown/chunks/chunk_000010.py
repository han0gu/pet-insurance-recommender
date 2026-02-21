from langchain_core.documents import Document

chunk = Document(
    page_content=('- 말합니다.\n'
 '제2관 보험금의 지급# 제 3조 (보험금의 지급사유)회사는 피보험자가 보험기간 중에 상해로 장해분류표([별표2] 참조. 이하 '
 '같습니다)에서\n'
 '정한 장해지급률이 80% 이상에 해당하는 장해상태가 되었을 때에는 최초 1회에 한하여\n'
 '보험증권에 기재된 보험가입금액을 상해 후유장해(80%이상)보험금으로 보험수익자에게\n'
 '지급합니다.<용어풀이># [장해지급률]질병이나 상해에 대하여 치유 후 남아있는 영구적인 장해에 의한 신체의 노동력 상실정도를 %로'),
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
