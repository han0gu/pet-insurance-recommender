from langchain_core.documents import Document

chunk = Document(
    page_content=('인해 발생하는 피신청인의 손해를 법적으로 보상해 주기 위해서 법원에 납부하\n'
 '는 공탁금을 대신하는 보험상품의 보험료를 말합니다.제4조(보상하지 않는 손해)회사는 아래의 사유를 원인으로 하여 생긴 배상책임을 '
 '부담함으로써 입은 손해는 보상하지\n'
 '않습니다.1. 계약자 또는 피보험자(법인인 경우에는 그 이사 또는 법인의 업무를 집행하는 그 밖의\n'
 '기관)또는 이들의 법정대리인의 고의\n'
 '2. 전쟁, 혁명, 내란, 사변, 테러, 폭동, 소요, 노동쟁의, 기타 이들과 유사한 사태\n'
 '3. 지진, 분화, 홍수, 해일 또는 이와 비슷한 천재지변'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
