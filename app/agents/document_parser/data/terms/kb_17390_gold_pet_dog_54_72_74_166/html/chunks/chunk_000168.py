from langchain_core.documents import Document

chunk = Document(
    page_content=(". 전화를 이용하여 청약내용, 보험료납입, 보험기간, 계약 전 알릴 의무, 약관의</p><br><p id='208' "
 "data-category='paragraph' style='font-size:16px'>중요한 내용 등 계약을 체결하는 데 필요한 사항을 "
 '질문 또는 설명하는 방법.<br>이 경우 계약자의 답변과 확인내용을 음성 녹음함으로써 약관의 중요한 내용을<br>설명한 것으로 '
 '봅니다.<br>\uf000 다음의 어느 하나의 경우 계약자는 계약이 성립한 날부터 3개월 이내에 계약을 취</p><br><p '
 "id='209'"),
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
