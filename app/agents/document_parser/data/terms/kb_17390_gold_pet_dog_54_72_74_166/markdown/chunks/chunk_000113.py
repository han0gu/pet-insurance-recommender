from langchain_core.documents import Document

chunk = Document(
    page_content=('이 경우 계약자의 답변과 확인내용을 음성 녹음함으로써 약관의 중요한 내용을\n'
 '설명한 것으로 봅니다.\n'
 '\uf000 다음의 어느 하나의 경우 계약자는 계약이 성립한 날부터 3개월 이내에 계약을 취소할 수 있습니다.\n'
 '1. 회사가 제1항에 따라 제공하여야 할 약관 및 계약자 보관용 청약서를 계약자가- 청약할 때 계약자에게 전달하지 않았거나 약관의 중요한 '
 '내용을 설명하지 않은\n'
 '- 경우\n'
 '- 2. 계약을 체결할 때 계약자가 청약서에 자필서명을 하지 않은 경우(자필서명에는\n'
 '도장을 찍는 날인과 전자서명법 제2조 제2호에 따른 전자서명을 포함합니다)'),
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
