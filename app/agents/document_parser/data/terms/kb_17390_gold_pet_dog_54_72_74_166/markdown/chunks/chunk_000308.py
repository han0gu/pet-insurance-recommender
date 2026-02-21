from langchain_core.documents import Document

chunk = Document(
    page_content=('약의 소멸) 및 제36조(중도인출)는 제외합니다.- 80 -# 8. 상해입원일당(1일이상)Ⅱ- 제1조(보험금의 지급사유)\n'
 '- \uf000 회사는 이 특별약관의 보험기간 중에 피보험자가 상해의 직접결과로써 생활기능\n'
 '- 또는 업무능력에 지장을 가져와 병원 또는 의원(한방병원 또는 한의원을 포함합\n'
 '- 니다)에 입원하여 치료를 받은 경우에는 최초 입원일로부터 입원 1일당 이 특별\n'
 '- 약관의 보험가입금액을 상해입원일당으로 보험수익자에게 지급합니다.'),
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
