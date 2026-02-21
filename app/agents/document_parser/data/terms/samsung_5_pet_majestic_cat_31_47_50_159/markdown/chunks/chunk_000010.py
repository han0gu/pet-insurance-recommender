from langchain_core.documents import Document

chunk = Document(
    page_content=('정한 장해지급률이 80% 이상에 해당하는 장해상태가 되었을 때에는 최초 1회에 한하여\n'
 '보험증권에 기재된 보험가입금액을 상해 후유장해(80%이상)보험금으로 보험수익자에게\n'
 '지급합니다.# <용어풀이># [장해지급률]질병이나 상해에 대하여 치유 후 남아있는 영구적인 장해에 의한 신체의 노동력 상실정도를 %로\n'
 '나타낸 것을 말합니다.# 제 4조 (보험금 지급에 관한 세부규정)- ① 제3조(보험금의 지급사유) 에서 장해지급률이 상해 발생일부터 '
 '180일 이내에 확정되'),
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
