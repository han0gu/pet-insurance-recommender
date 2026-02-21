from langchain_core.documents import Document

chunk = Document(
    page_content=('정의 및<br>진단확정)에서 정한 "호흡기관련질병"으로 진단확정되고 그 치료를 직접적인 목적으<br>로 수술을 받은 때에는 수술 1회당 '
 "이 특별약관의 보험가입금액을 호흡기관련질병</p><br><h1 id='226' style='font-size:14px'>수술비로 "
 "보험수익자에게 지급합니다.</h1><p id='227' data-category='paragraph' "
 "style='font-size:14px'>제2조(보험금 지급에 관한 세부규정)<br>보험수익자와 회사가 제1조(보험금의 지급사유)의 "
 '보험금 지급사유에 대해'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
