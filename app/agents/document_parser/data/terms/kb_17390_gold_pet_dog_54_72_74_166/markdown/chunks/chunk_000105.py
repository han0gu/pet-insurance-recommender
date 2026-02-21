from langchain_core.documents import Document

chunk = Document(
    page_content=('성 상품, 투자성 상품, 대출성 상품 또는 금융상품자문에 관한 계약의 청약을 한\n'
 '일반금융소비자는 다음 각 호의 구분에 따른 기간(거래 당사자 사이에 다음 각\n'
 '호의 기간보다 긴 기간으로 약정한 경우에는 그 기간) 내에 청약을 철회할 수 있# 다.1. 보장성 상품: 일반금융소비자가 「상법」 '
 '제640조에 따른 보험증권을 받은\n'
 '날부터 15일과 청약을 한 날부터 30일 중 먼저 도래하는 기간제2조(정의) 이 법에서 사용하는 용어의 뜻은 다음과 같다.9. '
 '“전문금융소비자”란 금융상품에 관한 전문성 또는 소유자산규모 등에 비'),
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
